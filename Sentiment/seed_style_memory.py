"""Utility to seed the style memory vector database with optional extra docs/exemplars.

Run directly to ensure the collection contains the base style guide. You can pass a folder
path to ingest additional .txt files as exemplars.
"""
import os
import glob
from typing import List

from style_memory import upsert_style_guide, _get_vectorstore


def ingest_folder(folder: str) -> int:
    vs = _get_vectorstore()
    paths: List[str] = glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True)
    count = 0
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                text = f.read()
            if text.strip():
                vs.add_texts([text], metadatas=[{"type": "user_exemplar", "path": p}])
                count += 1
        except Exception:
            pass
    vs.persist()
    return count


if __name__ == "__main__":
    upsert_style_guide()
    folder = os.environ.get("STYLE_EXEMPLARS_DIR")
    if folder and os.path.isdir(folder):
        n = ingest_folder(folder)
        print(f"Seeded base style and ingested {n} exemplar files from {folder}.")
    else:
        print("Seeded base style. Set STYLE_EXEMPLARS_DIR to ingest extra .txt exemplars.")
