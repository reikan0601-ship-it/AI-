import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import torch
import open_clip
from PIL import Image

DB_PATH = Path("db.sqlite")
UPLOAD_DIR = Path("static/uploads")

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

@torch.no_grad()
def get_embedding(model, preprocess, device, image_path: str) -> np.ndarray:
    image = preprocess(load_image(image_path)).unsqueeze(0).to(device)
    feat = model.encode_image(image)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu().numpy()

def ensure_dirs():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def save_image_to_uploads(src_path: Path) -> Path:
    # 同名衝突を避けるため、連番を付ける
    dst = UPLOAD_DIR / src_path.name
    if dst.exists():
        stem = src_path.stem
        suf = src_path.suffix
        i = 1
        while True:
            cand = UPLOAD_DIR / f"{stem}_{i}{suf}"
            if not cand.exists():
                dst = cand
                break
            i += 1
    shutil.copy2(src_path, dst)
    return dst

def insert_item(image_path: str, embedding: np.ndarray, is_favorite: int = 0):
    emb_json = json.dumps(embedding.tolist())

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO items (image_path, embedding_json, is_favorite) VALUES (?, ?, ?)",
        (image_path, emb_json, int(is_favorite)),
    )
    conn.commit()
    item_id = cur.lastrowid
    conn.close()
    return item_id

def main():
    # ここを自分の画像パスに変える（例：app/static/uploads/a.jpg でもOK）
    src_image = Path("app/static/uploads/a.jpg")
    is_favorite = 0  # お気に入りにしたいなら 1

    if not src_image.exists():
        raise FileNotFoundError(f"画像が見つかりません: {src_image}")

    ensure_dirs()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "ViT-B-32"
    pretrained = "laion2b_s34b_b79k"

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device).eval()

    saved_path = save_image_to_uploads(src_image)
    emb = get_embedding(model, preprocess, device, str(saved_path))

    item_id = insert_item(str(saved_path), emb, is_favorite=is_favorite)
    print("OK: saved item")
    print(" id:", item_id)
    print(" image_path:", saved_path)
    print(" embedding_len:", emb.shape[0])

if __name__ == "__main__":
    main()

def insert_item_simple(
    image_path: str,
    embedding_json: str,
    category=None,
    colors=None,
    style_tags=None,
    season=None,
    occasions=None,
    is_favorite: int = 0,
):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO items (
            image_path, embedding_json, category, colors, style_tags,
            season, occasions, is_favorite
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            image_path,
            embedding_json,
            category,
            colors,
            style_tags,
            season,
            occasions,
            int(is_favorite),
        ),
    )
    conn.commit()
    item_id = cur.lastrowid
    conn.close()
    return item_id

insert_item_full = insert_item_simple

