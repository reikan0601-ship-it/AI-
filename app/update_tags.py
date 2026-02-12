import argparse
import sqlite3
from pathlib import Path

DB_PATH = Path("db.sqlite")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("id", type=int)
    p.add_argument("--style", default="", help="例: cute,clean,cool")
    p.add_argument("--occasion", default="", help="例: date,school,casual")
    p.add_argument("--category", default="")
    p.add_argument("--season", default="", help="例: spring,summer,autumn,winter")
    args = p.parse_args()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        UPDATE items
        SET style_tags = ?, occasions = ?, category = ?, season = ?
        WHERE id = ?
    """, (args.style, args.occasion, args.category, args.season, args.id))
    conn.commit()
    conn.close()
    print("OK updated:", args.id)

if __name__ == "__main__":
    main()
# 