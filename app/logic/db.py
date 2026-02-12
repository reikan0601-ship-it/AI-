import sqlite3
from pathlib import Path

DB_PATH = Path("db.sqlite")

def init_db():
    """Initialize the database"""
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
    print(f"OK: initialized {DB_PATH}")

def get_all_items():
    """Get all items from database"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    rows = cur.execute("SELECT * FROM items ORDER BY id DESC").fetchall()
    conn.close()
    return rows

def get_item(item_id: int):
    """Get a specific item by ID"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    row = cur.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
    conn.close()
    return row

def toggle_favorite(item_id: int, is_favorite: int = 1):
    """Toggle favorite status"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE items SET is_favorite = ? WHERE id = ?", (int(is_favorite), int(item_id)))
    conn.commit()
    conn.close()
    print(f"OK: item {item_id} is_favorite -> {is_favorite}")
