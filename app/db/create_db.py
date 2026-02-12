import sqlite3
from pathlib import Path

DB_PATH = Path("db.sqlite")

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_path TEXT NOT NULL,
        embedding_json TEXT NOT NULL,
        category TEXT,
        colors TEXT,
        style_tags TEXT,
        season TEXT,
        occasions TEXT,
        is_favorite INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """)

    conn.commit()
    conn.close()
    print(f"OK: created {DB_PATH}")

if __name__ == "__main__":
    main()
