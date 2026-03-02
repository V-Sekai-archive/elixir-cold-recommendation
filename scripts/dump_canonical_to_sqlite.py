#!/usr/bin/env python3
"""
Export canonical item texts from item_text_dict.pkl to SQLite so Elixir and Python
use the exact same input bytes. Uses the same formula as the RecGPT official script:
  str(info_dict[iid]).replace('{','').replace('}','')

Writes to the same DB/table that Elixir uses (canonical_item_texts). Run once after
fetching the pkl; then mix recgpt.build_fixture and mix recgpt.compare_embeddings
will read these bytes (default: --canonical-texts). No Python at runtime.

Usage:
  uv run python scripts/dump_canonical_to_sqlite.py --pkl data/steam/item_text_dict.pkl
  uv run python scripts/dump_canonical_to_sqlite.py --pkl data/steam/item_text_dict.pkl --db priv/recgpt.sqlite3

Set RECGPT_SQLITE_PATH to match Elixir's DB, or pass --db (default: priv/recgpt.sqlite3).
"""
import argparse
import pickle
import sqlite3
import os


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export RecGPT-official canonical item texts from pkl to SQLite (same as Elixir table)."
    )
    ap.add_argument("--pkl", default="data/steam/item_text_dict.pkl", help="Path to item_text_dict.pkl")
    ap.add_argument("--db", default=None, help="SQLite DB path (default: RECGPT_SQLITE_PATH or priv/recgpt.sqlite3)")
    ap.add_argument("--verify", action="store_true", help="After writing, verify first 5 rows match str(dict).replace from pkl")
    args = ap.parse_args()

    pkl_path = args.pkl
    db_path = args.db or os.environ.get("RECGPT_SQLITE_PATH", "priv/recgpt.sqlite3")

    if not os.path.isfile(pkl_path):
        raise SystemExit(f"pkl not found: {pkl_path}")

    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    with open(pkl_path, "rb") as f:
        info_dict = pickle.load(f)

    # Same order as official script: for iid in info_dict (insertion order in Python 3.7+)
    rows = []
    for iid in info_dict:
        s = str(info_dict[iid]).replace("{", "").replace("}", "")
        blob = s.encode("utf-8")
        rows.append((iid, blob))

    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS canonical_item_texts (item_id INTEGER PRIMARY KEY, text BLOB NOT NULL)"
    )
    conn.execute("DELETE FROM canonical_item_texts")
    conn.executemany("INSERT INTO canonical_item_texts (item_id, text) VALUES (?, ?)", rows)
    conn.commit()

    if args.verify:
        ok = True
        cursor = conn.execute(
            "SELECT item_id, text FROM canonical_item_texts ORDER BY item_id LIMIT 5"
        )
        for i, (iid, blob) in enumerate(cursor):
            expected = str(info_dict[iid]).replace("{", "").replace("}", "").encode("utf-8")
            if blob != expected:
                print(f"Verify FAIL row {i} item_id={iid}: SQLite blob != str(dict).replace")
                ok = False
            else:
                print(f"Verify OK row {i} item_id={iid} ({len(blob)} bytes)")
        if ok:
            print("All verified: SQLite contents match Python str(dict).replace.")

    conn.close()
    print(f"Wrote {len(rows)} rows to {db_path} (table canonical_item_texts).")
    print("Elixir build_fixture and compare_embeddings will use these bytes when canonical-texts is on (default).")


if __name__ == "__main__":
    main()
