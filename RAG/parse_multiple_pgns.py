# parse_multiple_pgns.py
# -*- coding: utf-8 -*-
import os
import glob
import inspect

import parse_single_pgn as parser

# ---- Constants (adjust if needed) ----
DATA_DIR = "./data"
DATABASE_DIR = "./database"
COLLECTION_NAME = getattr(parser, "COLLECTION_NAME", "chess_games")

essential_dirs = [DATA_DIR, DATABASE_DIR]
for d in essential_dirs:
    os.makedirs(d, exist_ok=True)

# ---- Index all PGNs ----
pgn_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pgn")))

for p in pgn_files:
    try:
        parser.process_pgn_file(pgn_path=p,
                                database_dir=DATABASE_DIR,
                                collection_name=COLLECTION_NAME)
        print(f"[{os.path.basename(p)}] added new embeddings")
    except Exception as e:
        print(f"[{os.path.basename(p)}] ERROR: {e}")
print(f"Chroma DB persisted at: {os.path.abspath(DATABASE_DIR)}")

col = parser.load_collection(DATABASE_DIR, collection_name=COLLECTION_NAME)
parser.demo_queries(col)