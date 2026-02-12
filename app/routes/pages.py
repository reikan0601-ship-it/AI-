import numpy as np
from app.logic.embeddings import get_image_embedding
from app.logic.scoring import score_candidate
from app.db.add_item import insert_item_simple as insert_item_full
from app.db.toggle_favorite import toggle_favorite



import json
import shutil
from pathlib import Path

from flask import Blueprint, render_template, request, redirect
from werkzeug.utils import secure_filename

from app.logic.clip import get_tag_candidates
from app.db.list_items import list_items


bp = Blueprint("pages", __name__)

UPLOAD_DIR = Path("app/static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CLOSET_DIR = Path("app/static/closet")
CLOSET_DIR.mkdir(parents=True, exist_ok=True)


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/judge", methods=["POST"])
def judge():
    files = [request.files.get("img1"), request.files.get("img2"), request.files.get("img3")]
    if any(f is None or f.filename == "" for f in files):
        return "画像が3枚必要です。戻って選び直してね。", 400

    results = []

    occasion = (request.form.get("occasion") or "").strip().lower()
    closet_items = list_items()
    for f in files:
        name = secure_filename(f.filename)
        save_path = UPLOAD_DIR / name

        # 同名ファイル対策（_1, _2...）
        if save_path.exists():
            stem = save_path.stem
            suf = save_path.suffix
            i = 1
            while True:
                cand = UPLOAD_DIR / f"{stem}_{i}{suf}"
                if not cand.exists():
                    save_path = cand
                    break
                i += 1

        f.save(save_path)

        # タグ候補（上位3つ）
        tags = get_tag_candidates(str(save_path), topk=3)

        # テンプレで扱いやすい形に寄せる
        norm_tags = []
        for t in tags:
            if isinstance(t, dict):
                norm_tags.append({"tag": t.get("tag"), "score": t.get("score")})
            else:
                norm_tags.append({"tag": getattr(t, "tag", str(t)), "score": getattr(t, "score", None)})

        # img表示用：/static/uploads/... を作る
        web_path = f"static/uploads/{save_path.name}"

        # ===== embedding 計算 =====
        cand_emb = get_image_embedding(str(save_path))

        # L2 normalize（念のため）
        cand_emb = cand_emb.astype(np.float32)
        n = float(np.linalg.norm(cand_emb))
        if n != 0.0:
            cand_emb = cand_emb / n
# ===== スコア計算 =====
        sc = score_candidate(
            cand_emb=cand_emb,
            cand_tags=norm_tags,
            user_occasion=occasion,
            closet_items=closet_items,
        )
        results.append({
            "path": web_path,
            "tags": norm_tags,
            "score": sc
        })

    best = max(results, key=lambda r: r["score"]["final"])

    return render_template(
        "result.html",
        results=results,
        best=best,
        occasion=occasion,
        stdout="",
        stderr=""
        )


@bp.route("/buy", methods=["POST"])
def buy():
    rel_path = request.form.get("path")           # 例: static/uploads/xxx.jpg
    tags_json = request.form.get("tags_json", "[]")

    # ここ追加：中身確認（本番では消してOK）
    if not rel_path:
        return "path が空です", 400
    if not tags_json:
        return "tags_json が空です", 400
    
    # 安全チェック：uploads配下しか許可しない
    if not rel_path or not rel_path.startswith("static/uploads/"):
        return "不正なパスです", 400

    filename = Path(rel_path).name
    src = UPLOAD_DIR / filename

    if not src.exists():
        return "ファイルが見つかりません", 404

    # 移動先（同名対策）
    dst = CLOSET_DIR / src.name
    if dst.exists():
        stem, suf = dst.stem, dst.suffix
        i = 1
        while True:
            cand = CLOSET_DIR / f"{stem}_{i}{suf}"
            if not cand.exists():
                dst = cand
                break
            i += 1

    shutil.move(str(src), str(dst))

    tags = json.loads(tags_json)
    style_tags = ", ".join([t.get("tag") for t in tags if isinstance(t, dict) and t.get("tag")])

    # ✅ 本物embeddingを計算（モデルは1回ロードしたものを使い回す）
    emb = get_image_embedding(str(dst))                # np.ndarray
    emb_json = json.dumps(emb.tolist(), ensure_ascii=False)

    # ✅ DB本登録
    final_web_path = f"closet/{dst.name}"
    occasion = (request.form.get("occasion") or "").strip().lower() or None

    insert_item_full(
        image_path=final_web_path,
        embedding_json=emb_json,
        style_tags=style_tags,
        is_favorite=0,
    )

    return redirect("/closet")

@bp.route("/closet")
def closet():
    items = list_items()
    return render_template("closet.html", items=items)

@bp.route("/favorite/toggle", methods=["POST"])
def favorite_toggle():
    item_id = request.form.get("id")
    if not item_id:
        return "idがありません", 400

    toggle_favorite(int(item_id))
    return redirect("/closet")


