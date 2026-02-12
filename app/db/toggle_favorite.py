import sqlite3
from pathlib import Path

DB_PATH = Path("db.sqlite")

def set_favorite(item_id: int, is_favorite: int) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "UPDATE items SET is_favorite = ? WHERE id = ?",
        (int(is_favorite), int(item_id)),
    )
    conn.commit()
    conn.close()

def toggle_favorite(item_id: int) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE items SET is_favorite = 1 - is_favorite WHERE id = ?", (int(item_id),))
    conn.commit()
    conn.close()
