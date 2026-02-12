import sqlite3

DB_PATH = "db.sqlite"

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 1) "static/" で始まるものは "static/" を外す
    #    static/closet/xxx.jpg -> closet/xxx.jpg
    #    static/uploads/xxx.jpg -> uploads/xxx.jpg
    cur.execute("""
        UPDATE items
        SET image_path = SUBSTR(image_path, 8)
        WHERE image_path LIKE 'static/%'
    """)

    # 2) バックスラッシュをスラッシュに
    #    uploads\\a.jpg -> uploads/a.jpg
    cur.execute(r"""
        UPDATE items
        SET image_path = REPLACE(image_path, '\', '/')
        WHERE image_path LIKE '%\%'
    """)

    conn.commit()

    # 確認出力（任意）
    rows = cur.execute("SELECT id, image_path FROM items ORDER BY id DESC").fetchall()
    conn.close()

    print("OK: normalized image_path")
    for rid, p in rows:
        print(rid, p)

if __name__ == "__main__":
    main()
