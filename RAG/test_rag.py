import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parse_single_pgn import load_collection
from chromadb.api.models.Collection import Collection
from utility import fen_to_string

def demo_queries(collection: Collection):
    """
    Show simple examples using **UCI-labeled** documents:
    - Filter by winner
    - Similarity search with a filter
    """
    print("\n=== DEMO: similarity search over fen_to_string docs (winner=white) ===")
    # Query using FEN-string embedding (start position)
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    q = fen_to_string(fen)
    res = collection.query(query_texts=[q], n_results=5, where={"winner": "white"})
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    for d, m in zip(docs, metas):
        print(f"\nChunk:\n{d}\nMeta: {m}")

if __name__ == "__main__":

    collection = load_collection(database_path="./database/")
    demo_queries(collection)