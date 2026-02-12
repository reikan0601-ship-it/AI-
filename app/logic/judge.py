import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import sys
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
import open_clip
from PIL import Image

DB_PATH = Path("db.sqlite")


# ===== Utilities =====

def _parse_tags(s):
    if not s:
        return set()
    return {t.strip() for t in s.split(",") if t.strip()}

def _jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)

def tag_score(candidate, closet_items, w_style=0.5, w_occ=0.3, w_season=0.2):
    c_style = _parse_tags(candidate.get("style", ""))
    c_occ = _parse_tags(candidate.get("occasion", ""))
    c_season = _parse_tags(candidate.get("season", ""))

    if not closet_items:
        return 1.0  # 服ゼロなら「被りなし」扱いで加点

    overlaps = []
    for it in closet_items:
        s = _jaccard(c_style, _parse_tags(it.get("style", "")))
        o = _jaccard(c_occ, _parse_tags(it.get("occasion", "")))
        se = _jaccard(c_season, _parse_tags(it.get("season", "")))
        overlaps.append(w_style*s + w_occ*o + w_season*se)

    avg_overlap = sum(overlaps) / len(overlaps)
    return 1.0 - avg_overlap  # 被り少ないほど高得点



def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # both must be L2-normalized
    return float(np.dot(a, b))


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


@torch.no_grad()
def get_embedding(model, preprocess, device, image_path: str) -> np.ndarray:
    img = preprocess(load_image(image_path)).unsqueeze(0).to(device)
    feat = model.encode_image(img)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu().numpy().astype(np.float32)


@dataclass
class ClosetItem:
    id: int
    image_path: str
    emb: np.ndarray
    is_favorite: int
    style: str
    occasion: str
    season: str


def load_closet_items(db_path: Path) -> List[ClosetItem]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT id, image_path, embedding_json, is_favorite, style, occasion, season FROM items"
    ).fetchall()
    conn.close()

    items: List[ClosetItem] = []
    for (item_id, image_path, emb_json, is_favorite, style, occasion, season
         ) in rows:
        emb_list = json.loads(emb_json)
        emb = np.array(emb_list, dtype=np.float32)
        # L2 normalize
        n = float(np.linalg.norm(emb))
        if n != 0.0:
            emb = emb / n

        items.append(
            ClosetItem(
                id=int(item_id),
                image_path=str(image_path),
                emb=emb,
                is_favorite=int(is_favorite),
                style=style or "",
                occasion=occasion or "",
                season=season or "",
            )
        )
    return items


def best_match(candidate_emb: np.ndarray, closet: List[ClosetItem]) -> Tuple[ClosetItem, float]:
    best_item = closet[0]
    best_sim = -1.0
    for it in closet:
        s = cosine_sim(candidate_emb, it.emb)
        if s > best_sim:
            best_sim = s
            best_item = it
    return best_item, best_sim


def main():
    p = argparse.ArgumentParser()
    p.add_argument("img1")
    p.add_argument("img2")
    p.add_argument("img3")
    p.add_argument("--buy_threshold", type=float, default=0.25, help="これ未満なら買わない")
    p.add_argument("--gap", type=float, default=0.12, help="1位と2位の差がこれ未満なら買わない")
    args = p.parse_args()

    closet = load_closet_items(DB_PATH)
    if len(closet) == 0:
        raise RuntimeError("DBに手持ち服がありません。先に add_item.py で items を追加してね。")

    # Load model once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "ViT-B-32"
    pretrained = "laion2b_s34b_b79k"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()

    candidates = [args.img1, args.img2, args.img3]
    cand_embs = [get_embedding(model, preprocess, device, path) for path in candidates]

    # ===== Scoring weights (TAG-LESS MVP) =====
    W_NOVELTY = 1.5           # 新規性（被りの少なさ）重視
    FAVORITE_SIM_TH = 0.80    # お気に入りに「十分似てる」とみなす閾値
    FAVORITE_BONUS = 0.6      # お気に入りに似てたら加点
    SAME_ITEM_TH = 0.90       # ほぼ同じ服の閾値
    SAME_ITEM_PENALTY = 1.0   # ほぼ同じなら強い減点

    results = []
    for idx, (path, emb) in enumerate(zip(candidates, cand_embs), start=1):
        # L2 normalize candidate embedding
        emb = emb.astype(np.float32)
        n = float(np.linalg.norm(emb))
        if n != 0.0:
            emb = emb / n

        bm_item, bm_sim = best_match(emb, closet)

        # ===== TAG-LESS scoring =====
        reason_bits = []

        novelty = 1.0 - bm_sim
        base_score = W_NOVELTY * novelty
        score = base_score
        reason_bits.append(f"新規性 = 1 - sim = {novelty:.3f}（sim={bm_sim:.3f}）")

        # ===== タグ加点 =====
        TAG_WEIGHT = 0.25

        candidate_dict = {
            "style": "",
            "occasion": "",
            "season": "",
        }

        closet_dicts = [
            {"style": it.style, "occasion": it.occasion, "season": it.season}
            for it in closet
        ]

        tag_bonus = tag_score(candidate_dict, closet_dicts)
        score += TAG_WEIGHT * tag_bonus
        reason_bits.append(f"タグ多様性ボーナス = {TAG_WEIGHT * tag_bonus:.3f}")

        # ほぼ同じ服を持ってるなら強い減点（favoriteでも例外なし）
        if bm_sim >= SAME_ITEM_TH:
            score -= SAME_ITEM_PENALTY
            reason_bits.append(f"似すぎ判定(sim≥{SAME_ITEM_TH}) → 強い減点 -{SAME_ITEM_PENALTY}")

        # お気に入り特例：お気に入りに似てて、かつ「似すぎ」じゃないなら加点
        if bm_item.is_favorite == 1 and (bm_sim >= FAVORITE_SIM_TH) and (bm_sim < SAME_ITEM_TH):
            score += FAVORITE_BONUS
            reason_bits.append(
                f"一番似てる手持ち(id={bm_item.id})がお気に入り＆十分似てる(sim≥{FAVORITE_SIM_TH}) → +{FAVORITE_BONUS}"
            )
        elif bm_item.is_favorite == 1:
            reason_bits.append("一番似てる手持ちはお気に入り（ただし加点条件は未達）")
        else:
            reason_bits.append("一番似てる手持ちは通常アイテム")
                # ほぼ同じ服を持ってるなら強い減点（favoriteでも例外なし）
        if bm_sim >= SAME_ITEM_TH:
                    score -= SAME_ITEM_PENALTY
                    reason_bits.append(f"似すぎ判定(sim≥{SAME_ITEM_TH}) → 強い減点 -{SAME_ITEM_PENALTY}")

                # お気に入り特例：お気に入りに似てて、かつ「似すぎ」じゃないなら加点
        if bm_item.is_favorite == 1 and (bm_sim >= FAVORITE_SIM_TH) and (bm_sim < SAME_ITEM_TH):
                    score += FAVORITE_BONUS
                    reason_bits.append(
                        f"一番似てる手持ち(id={bm_item.id})がお気に入り＆十分似てる(sim≥{FAVORITE_SIM_TH}) → +{FAVORITE_BONUS}"
                    )
        elif bm_item.is_favorite == 1:
                    reason_bits.append("一番似てる手持ちはお気に入り（ただし加点条件は未達）")
        else:
                    reason_bits.append("一番似てる手持ちは通常アイテム")

        results.append(
                    {
                        "idx": idx,
                        "path": path,
                        "score": float(score),
                        "best_sim": float(bm_sim),
                        "best_item_id": bm_item.id,
                        "best_item_path": bm_item.image_path,
                        "best_item_fav": bm_item.is_favorite,
                        "reasons": reason_bits,
                    }
                )

            # sort by score desc
                # sort by score desc
    results.sort(key=lambda r: r["score"], reverse=True)

    # Decide buy / not buy
    top = results[0]
    second = results[1] if len(results) > 1 else {"score": 0.0}
    buy = True
    if top["score"] < args.buy_threshold:
        buy = False
    if (top["score"] - second["score"]) < args.gap:
        buy = False

    # ===== Print =====
    print("\n=== 判定結果（コンソール版 / タグ無し）===\n")
    for r in results:
        print(f"[候補{r['idx']}] {r['path']}")
        print(f"  score: {r['score']:.3f}")
        print(f"  best_match: id={r['best_item_id']} fav={r['best_item_fav']} path={r['best_item_path']}")
        for b in r["reasons"]:
            print(f"   - {b}")
        print()

    if buy:
        print(f"✅ おすすめ：候補{top['idx']}（score={top['score']:.3f}）")
        print(f"理由：スコアが最も高く、2位との差も十分（差={(top['score']-second['score']):.3f}）")
    else:
        print("❌ 今回は買わない（決め手が弱い）")
        print(f"  top_score={top['score']:.3f}, gap(top-2nd)={(top['score']-second['score']):.3f}")
        print("  ※閾値は --buy_threshold や --gap で調整できるよ")

    print("\n（調整）W_NOVELTY / FAVORITE_SIM_TH / SAME_ITEM_TH を変えると判定が変わるよ\n")


if __name__ == "__main__":
    main()
