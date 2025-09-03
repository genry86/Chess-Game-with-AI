import chess
from chromadb.api.models.Collection import Collection
import os
import chromadb
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from constants import FilePathConstants, DatabaseConstants, AIConstants
from typing import Dict, List, Tuple
import pathlib
import re

load_dotenv()

UCI_RE = re.compile(r'<final_move>\s*([a-h][1-8][a-h][1-8][qrbn]?|----)\s*</final_move>\s*\Z')
FILES = "abcdefgh"
FILE_TO_IDX = {f: i for i, f in enumerate(FILES)}
rag = None

# --- Public ---

def read_file(path):
    with open(path, "r") as f:
        return f.read()

def display_graph(graph, name):
    png_bytes = graph.get_graph().draw_mermaid_png()
    path = pathlib.Path(f"{name}.png")
    with open(path, "wb") as f:
        f.write(png_bytes)

def extract_final_move(text: str) -> str | None:
    m = UCI_RE.search(text.strip())
    return m.group(1) if m else None

def load_rag():
    global rag
    if rag is None:
        rag = load_collection()

def is_legal_step(fen: str, uci: str) -> bool:
    """
    Using current board fen, check if `uci` is legal step or not.
    """
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)

        return board.is_legal(move)
    except ValueError as e:
        return False

def apply_uci_move(fen: str, uci: str) -> str:
    """
    Applies a UCI move to a fen position and returns a new FEN.
    Supports captures, castling, en passant, promotions (e.g. 'e7e8q').
    """
    try:
        board = chess.Board(fen)
    except ValueError as e:
        raise ValueError(f"Invalid FEN: {e}")

    try:
        move = chess.Move.from_uci(uci)
    except ValueError as e:
        raise ValueError(f"Bad UCI format '{uci}': {e}")

    if not board.is_legal(move):
        # parse_san is not needed here: we already have UCI
        raise ValueError(f"Illegal move '{uci}' for given position.")

    board.push(move)
    return board.fen()

def is_valid_fen(fen: str) -> bool:
    try:
        board = chess.Board(fen)
        return True
    except ValueError:
        return False

def is_legal_fen_transition(prev_fen: str, next_fen: str) -> bool:
    try:
        prev = chess.Board(prev_fen)
        next_board = chess.Board(next_fen)
    except ValueError:
        return False  # one of the FENs is invalid

    for move in prev.legal_moves:
        prev.push(move)
        same_position = (
            prev.board_fen() == next_board.board_fen()
            and prev.turn == next_board.turn
            and prev.castling_rights == next_board.castling_rights
        )
        prev.pop()
        if same_position:
            return True
    return False

def find_possible_steps(fen: str, ai_color: str):
    """
    Use RAG system to find possible moves for AI predicted steps node.
    Searches chess game database for similar board states.
    """
    # Create query from steps
    current_fen = fen
    current_board_description = fen_to_string(current_fen)
    query = current_board_description

    # Set filter based on AI color
    filter_condition = {"winner": ai_color}

    load_rag()

    # Query the collection
    result = rag.query(
        query_texts=[query],
        n_results=AIConstants.RAG_RESULTS_COUNT,
        where=filter_condition
    )

    metadatas = result.get("metadatas", [[]])[0]

    possible_steps = []
    for meta in metadatas:
        if "next_move" in meta:
            move = meta["next_move"]
            found = next((item for item in possible_steps if item["next_move"] == move), None)
            if found:
                found["score"] += 1
            else:
                possible_steps.append({
                    "score": 1,
                    "next_move": move
                })

    return possible_steps

def fen_to_json(fen: str, exclude_castling: bool = True) -> Dict[str, Dict[str, Dict]]:
    """
    Given:
      - fen: current position FEN,
    Replace each piece's `possible_steps` with ONLY those target squares that are
    **legal for that side to move** in python-chess (considering blockers, checks, etc.).
    If `exclude_castling=True`, castling is not considered.

    Returns a new structure with filtered `possible_steps`.
    """
    pieces_json = _fen_to_json_with_possible_steps(fen)

    base = chess.Board(fen)

    def collect_legal_targets_for_square(
        side_is_white: bool, from_sq_str: str
    ) -> List[str]:
        # точная копия, чтобы не трогать базовую
        b = chess.Board(base.fen())
        b.turn = chess.WHITE if side_is_white else chess.BLACK
        src = chess.parse_square(from_sq_str)

        dests: List[str] = []
        for mv in b.legal_moves:
            if mv.from_square != src:
                continue
            if exclude_castling and b.is_castling(mv):
                continue
            dests.append(_sq_to_str(mv.to_square))

        # убрать дубликаты (например, несколько промоций в один квадрат) и отсортировать
        return _sort_squares(list(set(dests)))

    # скопируем структуру и пройдём по каждому слоту
    out: Dict[str, Dict[str, Dict]] = {"white": {}, "black": {}}
    for side_key in ("white", "black"):
        side_is_white = (side_key == "white")
        new_side: Dict[str, Dict] = {}
        for slot_name, info in pieces_json.get(side_key, {}).items():
            ptype = info.get("type")
            pos = info.get("position")
            steps = info.get("possible_steps", []) or []

            # если позиция невалидна, просто прокинем пусто
            if not ptype or not pos or len(pos) < 2:
                new_side[slot_name] = {**info, "possible_steps": []}
                continue

            legal_targets = set(collect_legal_targets_for_square(side_is_white, pos))
            # оставляем только те, что реально легальны для python-chess
            filtered = [s for s in steps if s in legal_targets]
            new_side[slot_name] = {**info, "possible_steps": filtered}
        out[side_key] = new_side

    return out

def fen_to_string(fen: str) -> str:
    """
    Convert FEN to a fixed-slot, human-readable piece listing that ALWAYS uses the same
    number of slots per side and piece type:
      King: 1, Queen: 1, Rooks: 2, Bishops: 2, Knights: 2, Pawns: 8.
    Missing pieces are filled with "__" to keep line lengths constant.

    Output format (exactly):
    White pieces:
    King: <sq_or__>
    Queen: <sq_or__>
    Rooks: <sq_or__>, <sq_or__>
    Bishops: <sq_or__>, <sq_or__>
    Knights: <sq_or__>, <sq_or__>
    Pawns: <sq_or__>, <sq_or__>, <sq_or__>, <sq_or__>, <sq_or__>, <sq_or__>, <sq_or__>, <sq_or__>
    Black pieces:
    King: <...>
    ...
    """
    board_field = fen.split()[0]
    ranks = board_field.split("/")
    if len(ranks) != 8:
        raise ValueError("Invalid FEN: board layout must have 8 ranks")

    acc: Dict[str, Dict[str, List[str]]] = {
        "white": {"K": [], "Q": [], "R": [], "B": [], "N": [], "P": []},
        "black": {"K": [], "Q": [], "R": [], "B": [], "N": [], "P": []},
    }

    FILES = "abcdefgh"

    # Parse board into square lists
    for r_idx, row in enumerate(ranks):
        rank_num = 8 - r_idx
        f_idx = 0
        for ch in row:
            if ch.isdigit():
                f_idx += int(ch)
                continue
            if f_idx >= 8:
                raise ValueError("Invalid FEN: file index out of range")
            sq = f"{FILES[f_idx]}{rank_num}"
            side = "white" if ch.isupper() else "black"
            piece = ch.upper()
            if piece in acc[side]:
                acc[side][piece].append(sq)
            # ignore unknown letters (variants)
            f_idx += 1

    # Helpers
    def file_index(sq: str) -> int:
        return FILES.index(sq[0])

    def rank_index(sq: str) -> int:
        return int(sq[1:])

    def sort_for_slots(side: str, squares: List[str]) -> List[str]:
        # Prefer “advanced” pieces (toward opponent), then by file a→h
        if side == "white":
            # higher rank first (8→1)
            return sorted(squares, key=lambda s: (-rank_index(s), file_index(s)))
        else:
            # black: lower rank first (1←8)
            return sorted(squares, key=lambda s: (rank_index(s), file_index(s)))

    def take_slots(side: str, piece: str, want: int) -> List[str]:
        squares = sort_for_slots(side, acc[side][piece])
        chosen = squares[:want]
        if len(chosen) < want:
            chosen += ["__"] * (want - len(chosen))
        return chosen

    def line(label: str, slots: List[str]) -> str:
        return f"{label}: " + ", ".join(slots)

    # Build output
    out_lines: List[str] = []

    for side in ("white", "black"):
        out_lines.append(f"{side.capitalize()} pieces:")
        out_lines.append(line("King",    take_slots(side, "K", 1)))
        out_lines.append(line("Queen",   take_slots(side, "Q", 1)))
        out_lines.append(line("Rooks",   take_slots(side, "R", 2)))
        out_lines.append(line("Bishops", take_slots(side, "B", 2)))
        out_lines.append(line("Knights", take_slots(side, "N", 2)))
        out_lines.append(line("Pawns",   take_slots(side, "P", 8)))

    return "\n".join(out_lines)

def load_collection(database_path: str = FilePathConstants.DEFAULT_DATABASE_PATH, collection_name: str = DatabaseConstants.DEFAULT_COLLECTION_NAME) -> Collection:
    """
    Load an existing Chroma collection (or create it if missing) WITHOUT reindexing PGN.
    Useful for quick experiments and queries.
    """
    data_dir = os.path.abspath(database_path)
    persist_dir = os.path.join(data_dir, FilePathConstants.DATABASE_SUBDIRECTORY)
    os.makedirs(persist_dir, exist_ok=True)
    col = get_or_create_collection(persist_dir, collection_name)
    print(f"Loaded Chroma collection at: {persist_dir}")
    return col

def get_or_create_collection(persist_dir: str, name: str) -> Collection:
    """Create or load Chroma collection with selected embedding function."""
    client: PersistentClient = chromadb.PersistentClient(path=persist_dir)
    ef = _make_embedding_function()
    try:
        col = client.get_collection(name=name, embedding_function=ef)
    except Exception as e:
        col = client.create_collection(name=name, embedding_function=ef)
    return col

# --- Private ---

def _make_embedding_function():
    """
    Prefer OpenAI text-embedding-3-large, with graceful fallback.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        # text-embedding-3-large: high quality, 3072-dim, robust multilingual support
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-large",
        )
    # Fallback for offline/dev: small, fast local model
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

def _fen_to_json_description(fen: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Convert FEN into a JSON-like description of pieces.
    """
    board_field = fen.split()[0]
    ranks = board_field.split("/")
    if len(ranks) != 8:
        raise ValueError("Invalid FEN: board layout must have 8 ranks")

    acc: Dict[str, Dict[str, List[str]]] = {
        "white": {"K": [], "Q": [], "R": [], "B": [], "N": [], "P": []},
        "black": {"K": [], "Q": [], "R": [], "B": [], "N": [], "P": []},
    }

    # --- parse board into square lists ---
    for r_idx, row in enumerate(ranks):
        rank_num = 8 - r_idx
        f_idx = 0
        for ch in row:
            if ch.isdigit():
                f_idx += int(ch)
                continue
            if f_idx >= 8:
                raise ValueError("Invalid FEN: file index out of range")
            sq = f"{FILES[f_idx]}{rank_num}"
            side = "white" if ch.isupper() else "black"
            piece = ch.upper()
            if piece in acc[side]:
                acc[side][piece].append(sq)
            f_idx += 1

    # --- helpers for deterministic ordering & slot selection ---
    def file_index(sq: str) -> int:
        return FILES.index(sq[0])

    def rank_index(sq: str) -> int:
        return int(sq[1:])

    def sort_for_slots(side: str, squares: List[str]) -> List[str]:
        if side == "white":
            return sorted(squares, key=lambda s: (-rank_index(s), file_index(s)))
        else:
            return sorted(squares, key=lambda s: (rank_index(s), file_index(s)))

    def take_slots(side: str, piece_letter: str, want: int) -> List[str]:
        squares = sort_for_slots(side, acc[side][piece_letter])
        chosen = squares[:want]
        if len(chosen) < want:
            chosen += ["__"] * (want - len(chosen))
        return chosen

    # --- build side json with exact keys ---
    def build_side(side: str) -> Dict[str, Dict[str, str]]:
        plan = [
            (["King"],               "K", 1, "king"),
            (["Queen"],              "Q", 1, "queen"),
            (["Rook1", "Rook2"],     "R", 2, "rook"),
            (["Bishop1", "Bishop2"], "B", 2, "bishop"),
            (["Knight1", "Knight2"], "N", 2, "knight"),
            (["Pawn1","Pawn2","Pawn3","Pawn4","Pawn5","Pawn6","Pawn7","Pawn8"], "P", 8, "pawn"),
        ]
        out: Dict[str, Dict[str, str]] = {}
        for labels, letter, count, type_name in plan:
            slots = take_slots(side, letter, count)
            for name, pos in zip(labels, slots):
                if pos != "__":  # <---- убираем пустые
                    out[name] = {"type": type_name, "position": pos}
        return out

    return {
        "white": build_side("white"),
        "black": build_side("black"),
    }

def _in_board(f: int, r: int) -> bool:
    return 0 <= f < 8 and 1 <= r <= 8

def _sq_to_fr(sq: str) -> Tuple[int, int]:
    """Convert 'e4' -> (file_idx, rank). file_idx in 0..7, rank in 1..8"""
    return FILE_TO_IDX[sq[0]], int(sq[1])

def _fr_to_sq(f: int, r: int) -> str:
    return f"{FILES[f]}{r}"

def _ray_moves(file_idx: int, rank: int, directions: List[Tuple[int, int]]) -> List[str]:
    """Generate all squares along rays until board edge (no blockers)."""
    out: List[str] = []
    for df, dr in directions:
        f, r = file_idx + df, rank + dr
        while _in_board(f, r):
            out.append(_fr_to_sq(f, r))
            f += df
            r += dr
    return out

def _king_moves(file_idx: int, rank: int) -> List[str]:
    out: List[str] = []
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == 0 and dr == 0:
                continue
            f, r = file_idx + df, rank + dr
            if _in_board(f, r):
                out.append(_fr_to_sq(f, r))
    return out

def _knight_moves(file_idx: int, rank: int) -> List[str]:
    out: List[str] = []
    for df, dr in ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)):
        f, r = file_idx + df, rank + dr
        if _in_board(f, r):
            out.append(_fr_to_sq(f, r))
    return out

def _pawn_moves(file_idx: int, rank: int, is_white: bool) -> List[str]:
    """
    Geometric pawn pattern ignoring blockers:
    - Forward 1
    - Forward 2 only from start rank (white: 2 -> 4, black: 7 -> 5)
    - Captures: diagonals one step forward
    """
    out: List[str] = []
    if is_white:
        # forward
        if _in_board(file_idx, rank + 1):
            out.append(_fr_to_sq(file_idx, rank + 1))
        if rank == 2 and _in_board(file_idx, 4):
            out.append(_fr_to_sq(file_idx, 4))
        # captures
        for df in (-1, 1):
            f, r = file_idx + df, rank + 1
            if _in_board(f, r):
                out.append(_fr_to_sq(f, r))
    else:
        # forward
        if _in_board(file_idx, rank - 1):
            out.append(_fr_to_sq(file_idx, rank - 1))
        if rank == 7 and _in_board(file_idx, 5):
            out.append(_fr_to_sq(file_idx, 5))
        # captures
        for df in (-1, 1):
            f, r = file_idx + df, rank - 1
            if _in_board(f, r):
                out.append(_fr_to_sq(f, r))
    # Deduplicate + sort by file then rank for stable output
    out = sorted(set(out), key=lambda s: (FILE_TO_IDX[s[0]], int(s[1])))
    return out

def _possible_steps_for(piece_type: str, pos: str, is_white: bool) -> List[str]:
    """Return all board-edge-limited geometric targets for the given piece."""
    f, r = _sq_to_fr(pos)

    if piece_type == "king":
        return _king_moves(f, r)

    if piece_type == "queen":
        # queen = rook rays + bishop rays
        rook_dirs = [(1,0), (-1,0), (0,1), (0,-1)]
        bishop_dirs = [(1,1), (1,-1), (-1,1), (-1,-1)]
        moves = _ray_moves(f, r, rook_dirs) + _ray_moves(f, r, bishop_dirs)
        return sorted(set(moves), key=lambda s: (FILE_TO_IDX[s[0]], int(s[1])))

    if piece_type == "rook":
        dirs = [(1,0), (-1,0), (0,1), (0,-1)]
        moves = _ray_moves(f, r, dirs)
        return sorted(set(moves), key=lambda s: (FILE_TO_IDX[s[0]], int(s[1])))

    if piece_type == "bishop":
        dirs = [(1,1), (1,-1), (-1,1), (-1,-1)]
        moves = _ray_moves(f, r, dirs)
        return sorted(set(moves), key=lambda s: (FILE_TO_IDX[s[0]], int(s[1])))

    if piece_type == "knight":
        return _knight_moves(f, r)

    if piece_type == "pawn":
        return _pawn_moves(f, r, is_white)

    # Unknown piece types (for variants) -> no steps
    return []

def _fen_to_json_with_possible_steps(fen: str) -> Dict[str, Dict[str, Dict]]:
    """
    Using `_fen_to_json_description(fen)` as source (which returns:
      { "white": {<SlotName>: {"type": "...", "position": "e4"}, ...},
        "black": {...} }
    ),
    enrich each piece object with `possible_steps`: a list of squares that the piece could
    geometrically move to (board-edge limited), ignoring blockers and chess legality.

    Returns the same structure but with an extra key per piece:
      {
        "white": {
          "Knight1": {"type": "knight", "position": "c3", "possible_steps": [...]},
          ...
        },
        "black": { ... }
      }
    """
    data = _fen_to_json_description(fen)  # existing function you already have
    out: Dict[str, Dict[str, Dict]] = {"white": {}, "black": {}}

    for side_key in ("white", "black"):
        is_white = (side_key == "white")
        side_dict = data.get(side_key, {})
        new_side: Dict[str, Dict] = {}
        for slot_name, info in side_dict.items():
            piece_type = info.get("type")
            pos = info.get("position")
            if not piece_type or not pos or len(pos) < 2:
                # Skip malformed entries
                continue
            steps = _possible_steps_for(piece_type, pos, is_white)
            new_side[slot_name] = {
                "type": piece_type,
                "position": pos,
                "possible_steps": steps,
            }
        out[side_key] = new_side

    return out

def _sq_to_str(sq: int) -> str:
    return f"{FILES[chess.square_file(sq)]}{chess.square_rank(sq) + 1}"

def _sort_squares(squares: List[str]) -> List[str]:
    return sorted(squares, key=lambda s: (FILE_TO_IDX[s[0]], int(s[1:])))

if __name__ == "__main__":
    # prev_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    # next_fen = "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1"
    #
    # print(is_legal_fen_transition(prev_fen, next_fen))
    #
    # bad_fen = "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 1 1"
    # print(is_legal_fen_transition(prev_fen, bad_fen))

    print(fen_to_string("r1bqkb1r/pppp1ppp/2npp3/4p3/2B1P1b1/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 5"))
    filtered_json = fen_to_json("r1bqkb1r/pppp1ppp/2npp3/4p3/2B1P1b1/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 5")
    print(filtered_json)