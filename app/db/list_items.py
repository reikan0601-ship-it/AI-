import json
import sqlite3
from pathlib import Path

DB_PATH = Path("db.sqlite")


def list_items():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT * FROM items ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ↓ CLIで確認したいとき用（残してもOK）
def main():
    rows = list_items()

    if not rows:
        print("No items found.")
        return

    for row in rows[:10]:
        emb = json.loads(row["embedding_json"])
        print("-" * 40)
        print("id:", row["id"])
        print("image_path:", row["image_path"])
        print("is_favorite:", row["is_favorite"])
        print("style:", row["style_tags"])
        print("occasion:", row["occasions"])
        print("season:", row["season"])
        print("embedding_len:", len(emb))
        print("embedding_head:", emb[:5])


if __name__ == "__main__":
    main()
