# parser.py
# https://www.pgnmentor.com/files.html 40 first files
# -*- coding: utf-8 -*-
"""
PGN -> Chroma indexer for chess move prediction windows.

Features:
- Robust PGN parsing (tags + movetext separated by a blank line).
- Build ~12-full-move windows so that the "next move" belongs to the game winner.
- Skip draws.
- Persist Chroma DB locally next to the PGN file (./data/chroma).
- Deduplication: stable IDs via SHA1 => re-running won't duplicate entries.
- A loader to open an existing DB without reindexing.

Usage:
    python parser.py
"""

import os
import re
import uuid
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from chromadb.api.models.Collection import Collection
import chromadb
import chess

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utility import get_or_create_collection, load_collection, fen_to_string

# ---------------------------
# Parsing utilities
# ---------------------------

@dataclass
class FullMove:
    """Represents a full move: number + optional white and black SANs."""
    number: int
    white: Optional[str] = None
    black: Optional[str] = None


RESULT_MAP = {
    "1-0": "white",
    "0-1": "black",
    "1/2-1/2": "draw",
    "*": "unknown",
}

# ---------------------------
# UCI-labeled helpers
# ---------------------------

def _push_san_if(board: chess.Board, san: Optional[str]):
    if san:
        move = board.parse_san(san)
        board.push(move)
        return move
    return None

def build_uci_labeled_window(moves: List[FullMove], start_n: int, end_n: int, end_after_black: bool) -> Tuple[str, chess.Board]:
    """
    Build a labeled UCI string for the window [start_n..end_n].
    Returns (uci_labeled_string, board_after_window).
    The board is advanced from the initial position through all moves up to the end of the window,
    so it can be used to convert the next SAN move to UCI reliably.
    """
    board = chess.Board()
    # advance to the start of window
    for n in range(1, start_n):
        mv = moves[n-1]
        _push_san_if(board, mv.white)
        _push_san_if(board, mv.black)

    parts: List[str] = []
    for n in range(start_n, end_n + 1):
        mv = moves[n-1]
        # white move (if present)
        if mv.white:
            prefix = "white-" if board.turn == chess.WHITE else "black-"
            m = board.parse_san(mv.white)
            parts.append(prefix + m.uci())
            board.push(m)
        # black move (respect end_after_black rule)
        if mv.black and (n < end_n or end_after_black):
            prefix = "white-" if board.turn == chess.WHITE else "black-"
            m = board.parse_san(mv.black)
            parts.append(prefix + m.uci())
            board.push(m)

    return " ".join(parts).strip(), board

def san_next_to_uci(board: chess.Board, san: Optional[str]) -> Optional[str]:
    """Convert the next SAN move in this position to UCI (without prefix)."""
    if not san:
        return None
    m = board.parse_san(san)
    return m.uci()


# ---------------------------
# Winner halfmove sampler
# ---------------------------
from typing import List, Tuple
def build_winner_halfmove_samples(moves: List[FullMove], winner: str) -> List[Tuple[str, str, str]]:
    """
    Build samples for the winner's half-moves.
    Each sample is (fen_to_string(fen_before), next_uci, fen_before).
    For white: samples for each white move; for black: samples for each black move.
    """
    board = chess.Board()
    samples: List[Tuple[str, str, str]] = []
    if winner == "white":
        for n in range(1, len(moves) + 1):
            mv = moves[n-1]
            if mv.white:
                fen_before = board.fen()
                next_uci = san_next_to_uci(board, mv.white)
                if next_uci:
                    samples.append((fen_to_string(fen_before), next_uci, fen_before))
                board.push(board.parse_san(mv.white))
            if mv.black:
                board.push(board.parse_san(mv.black))
    elif winner == "black":
        for n in range(1, len(moves) + 1):
            mv = moves[n-1]
            if mv.white and mv.black:
                board.push(board.parse_san(mv.white))
                fen_before = board.fen()
                next_uci = san_next_to_uci(board, mv.black)
                if next_uci:
                    samples.append((fen_to_string(fen_before), next_uci, fen_before))
                board.push(board.parse_san(mv.black))
    return samples


def read_file(path: str) -> str:
    """Read UTF-8 text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_into_game_blocks(pgn_text: str) -> List[str]:
    """
    Split the PGN text into game blocks.
    We detect a new game when a line starts with '[' (tag pair).
    We collect lines until the next tag-start or EOF.
    """
    lines = pgn_text.splitlines()
    games: List[List[str]] = []
    cur: List[str] = []

    def flush():
        if cur and any(line.strip() for line in cur):
            games.append(cur.copy())

    for line in lines:
        if line.startswith("[") and line.endswith("]") and cur:
            # New tags -> new game begins, flush previous
            flush()
            cur = [line]
        else:
            cur.append(line)

    flush()
    return ["\n".join(g) for g in games]


def parse_tag_section(block: str) -> Dict[str, str]:
    """
    Parse PGN tag pairs like: [Result "1-0"]
    Returns dict of tags.
    """
    tags = {}
    for line in block.splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            m = re.match(r'^\[(\w+)\s+"(.*)"\]$', line)
            if m:
                tags[m.group(1)] = m.group(2)
    return tags


def extract_movetext(block: str) -> str:
    """
    Extract movetext (SAN sequence) from a game block.
    Movetext is typically after a blank line following tags.
    """
    parts = block.strip().split("\n\n", 1)
    if len(parts) == 1:
        # No blank separator found; try to find first non-tag line
        lines = block.splitlines()
        start_idx = 0
        for i, line in enumerate(lines):
            if not (line.strip().startswith("[") and line.strip().endswith("]")):
                start_idx = i
                break
        movetext = "\n".join(lines[start_idx:])
    else:
        movetext = parts[1]

    # Merge wrapped lines
    movetext = " ".join(line.strip() for line in movetext.splitlines())
    return movetext.strip()


def clean_movetext(movetext: str) -> str:
    """
    Remove comments {...}, NAGs $x, and variations (...) from movetext.
    Keep move numbers and SAN.
    """
    movetext = re.sub(r"\{[^}]*\}", " ", movetext)     # remove {...}
    movetext = re.sub(r"\$\d+", " ", movetext)         # remove $N
    movetext = re.sub(r"\([^)]*\)", " ", movetext)     # remove ( ... )
    movetext = re.sub(r"\s+", " ", movetext).strip()   # normalize spaces
    return movetext


def tokenize_movetext(movetext: str) -> List[str]:
    """Split movetext into tokens (numbers, SAN, results)."""
    return movetext.split()


def tokens_to_fullmoves(tokens: List[str]) -> Tuple[List[FullMove], Optional[str]]:
    """
    Convert tokens into a list of FullMove. Returns (moves, result_token).
    We interpret tokens like: "1.e4", "c5", "2.Nf3", "a6", ..., "1-0"
    """
    moves: List[FullMove] = []
    current_number = None
    expecting_black = False
    result_token = None

    def is_result(tok: str) -> bool:
        return tok in RESULT_MAP

    def is_move_number_with_dot(tok: str) -> Optional[int]:
        # e.g., "12.", "12..."
        m = re.match(r"^(\d+)\.(\.\.)?$", tok)
        if m:
            return int(m.group(1))
        # e.g., "12.e4" fused
        m2 = re.match(r"^(\d+)\.(\.\.)?([^\s]+)$", tok)
        if m2:
            return int(m2.group(1))
        return None

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if is_result(tok):
            result_token = tok
            break

        number_with_dot = is_move_number_with_dot(tok)
        if number_with_dot is not None:
            fused = re.match(r"^(\d+)\.(\.\.)?([^\s]+)?$", tok)
            current_number = number_with_dot
            if fused and fused.group(3):
                san = fused.group(3)
                after_dots = fused.group(2) == ".."
                # Ensure move entry exists
                while len(moves) < current_number:
                    moves.append(FullMove(number=len(moves)+1))
                if after_dots:
                    moves[current_number-1].black = san
                    expecting_black = False
                else:
                    moves[current_number-1].white = san
                    expecting_black = True
            else:
                expecting_black = False
            i += 1
            continue

        # Plain SAN token (white or black)
        if current_number is None:
            i += 1
            continue

        while len(moves) < current_number:
            moves.append(FullMove(number=len(moves)+1))

        if not expecting_black:
            if moves[current_number-1].white is None:
                moves[current_number-1].white = tok
            else:
                moves[current_number-1].black = tok
            expecting_black = True
        else:
            moves[current_number-1].black = tok
            expecting_black = False

        i += 1

    return moves, result_token


def render_chunk(moves: List[FullMove], start_n: int, end_n: int, end_after_black: bool) -> str:
    """
    (Deprecated) Kept for compatibility elsewhere. Not used for embeddings anymore.
    """
    parts: List[str] = []
    for n in range(start_n, end_n + 1):
        mv = moves[n-1]
        if mv.white:
            parts.append(f"{n}.{mv.white}")
        if mv.black:
            if n < end_n or end_after_black:
                parts.append(mv.black)
    return " ".join(parts).strip()

# ---------------------------
# Chroma utilities
# ---------------------------


def stable_id(file_prefix: str, game_index: int, chunk_index: int, winner: str, next_move: str, chunk: str) -> str:
    """
    Build a deterministic ID so repeated runs do not create duplicates.
    """
    key = f"{file_prefix}|g{game_index}|c{chunk_index}|{winner}|{next_move}|{chunk}".encode("utf-8")
    return hashlib.sha1(key).hexdigest()  # 40 hex chars

def normalize_san_window(text: str) -> str:
    """
    Remove move numbers like '21.' and compress whitespace in a SAN window.
    Keeps only the SAN tokens themselves so similarity search isn't biased
    by absolute move indices.
    """
    # remove e.g. "21.", "3.", including rare "21..." if present
    text = re.sub(r"\b\d+\.(?:\.\.)?", " ", text)
    # collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def add_windows_dedup(
    file_prefix: str,
    collection: Collection,
    game_index: int,
    windows: List[Tuple[str, str, str]],
    winner: str,
):
    """
    Add windows with deduplication by stable IDs.
    """
    if not windows:
        return 0

    # Prepare batch
    ids = []
    docs = []
    metas = []
    for i, (chunk, next_move, fen_before) in enumerate(windows):
        sid = stable_id(file_prefix, game_index, i, winner, next_move, chunk)
        ids.append(sid)
        docs.append(chunk)
        metas.append({
            "winner": winner,
            "next_move": next_move,
            "fen": fen_before,
        })

    # Check which IDs already exist
    existing = set()
    # Chroma supports get(ids=[...]) but typically in limited batch sizes; be safe with chunks
    B = 500
    for b in range(0, len(ids), B):
        batch_ids = ids[b:b+B]
        got = collection.get(ids=batch_ids)
        for eid in got.get("ids", []):
            existing.add(eid)

    # Filter to only new records
    new_ids, new_docs, new_metas = [], [], []
    for _id, d, m in zip(ids, docs, metas):
        if _id not in existing:
            new_ids.append(_id)
            new_docs.append(d)
            new_metas.append(m)

    if new_ids:
        collection.add(ids=new_ids, documents=new_docs, metadatas=new_metas)

    return len(new_ids)


# ---------------------------
# Public API
# ---------------------------

def process_pgn_file(
        pgn_path: str,
        database_dir: str,
        collection_name: str = "chess_games"
) -> Collection:
    """
    High-level entry point:
    - Parse games
    - Build windows
    - Index to Chroma with deduplication
    Returns the Chroma collection instance.
    """
    if not os.path.isfile(pgn_path):
        raise FileNotFoundError(f"PGN not found: {pgn_path}")

    data_dir = os.path.abspath(database_dir)
    persist_dir = os.path.join(data_dir, "chroma")
    os.makedirs(persist_dir, exist_ok=True)

    collection = get_or_create_collection(persist_dir, collection_name)

    text = read_file(pgn_path)
    blocks = split_into_game_blocks(text)

    added = 0
    skipped_games = 0

    for gi, block in enumerate(blocks):
        tags = parse_tag_section(block)
        movetext_raw = extract_movetext(block)
        movetext = clean_movetext(movetext_raw)
        tokens = tokenize_movetext(movetext)
        moves, result = tokens_to_fullmoves(tokens)

        result = result or tags.get("Result", "").strip()
        winner = RESULT_MAP.get(result, "unknown")

        if winner not in ("white", "black"):
            skipped_games += 1
            continue

        windows = build_winner_halfmove_samples(moves, winner=winner)
        added += add_windows_dedup(pgn_path, collection, gi, windows, winner)

    print(f"Added (new) windows: {added} | Skipped games: {skipped_games}")
    print(f"Chroma persisted at: {persist_dir}")
    return collection


# ---------------------------
# Demo queries
# ---------------------------

def demo_queries(collection: Collection):
    """
    Show simple examples using **UCI-labeled** documents:
    - Filter by winner
    - Similarity search with a filter
    """
    print("\n=== DEMO: similarity search over fen_to_string docs (winner=white) ===")
    # Query using FEN-string embedding (start position)
    board = chess.Board()
    q = fen_to_string(board.fen())
    res = collection.query(query_texts=[q], n_results=5, where={"winner": "white"})
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    for d, m in zip(docs, metas):
        print(f"\nChunk:\n{d}\nMeta: {m}")


# ---------------------------
# __main__
# ---------------------------

if __name__ == "__main__":
    # As requested
    pgn_path = "./data/Kasparov.pgn"
    database_path = "./database/"

    # 1) Index (with dedup) OR
    # collection = process_pgn_file(pgn_path, database_dir=database_path, collection_name="chess_games")

    # 2) (Alternative) Only load existing DB without reindexing:
    # Run small demo queries and print to console
    # collection = load_collection(database_path, collection_name="chess_games")
    # demo_queries(collection)