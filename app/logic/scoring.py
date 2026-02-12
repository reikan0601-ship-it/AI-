import json
import numpy as np

def _vec_from_json(emb_json: str) -> np.ndarray:
    v = np.array(json.loads(emb_json), dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n != 0.0:
        v = v / n
    return v

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    # both should be L2-normalized
    return float(np.dot(a, b))

def best_match(candidate: np.ndarray, closet_items: list[dict]) -> tuple[dict, float]:
    best_it = closet_items[0]
    best = -1.0
    for it in closet_items:
        ej = it.get("embedding_json")
        if not ej:
            continue
        v = _vec_from_json(ej)
        s = cos_sim(candidate, v)
        if s > best:
            best = s
            best_it = it
    return best_it, best

def parse_tags(s: str) -> set[str]:
    if not s:
        return set()
    return {t.strip().lower() for t in s.split(",") if t.strip()}

def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)

def scene_score(user_occasion: str, cand_tags: list[dict]) -> float:
    # cand_tags: [{"tag": "...", "score": 0.xx}, ...]
    if not user_occasion:
        return 0.0
    u = user_occasion.strip().lower()
    best = 0.0
    for t in cand_tags:
        if (t.get("tag") or "").strip().lower() == u:
            best = max(best, float(t.get("score") or 0.0))
    # 0..1に収める
    return max(0.0, min(1.0, best))

def score_candidate(
    cand_emb: np.ndarray,
    cand_tags: list[dict],
    user_occasion: str,
    closet_items: list[dict],
    # weights
    W_NOVELTY=1.2,
    W_SCENE=0.6,
    FAVORITE_SIM_TH=0.80,
    FAVORITE_BONUS=0.35,
    SAME_ITEM_TH=0.90,
    SAME_ITEM_PENALTY=0.9,
):
    """
    final = 新規性 + シーン一致 + (お気に入り寄せボーナス) - (似すぎペナルティ)
    """
    reasons = []

    bm_item, bm_sim = best_match(cand_emb, closet_items)

    # --- 新規性（かぶり度） ---
    novelty = 1.0 - bm_sim  # 似てないほど高い
    score = W_NOVELTY * novelty

    dup_pct = int(round(bm_sim * 100))
    nov_pct = int(round(novelty * 100))

    if bm_sim >= SAME_ITEM_TH:
        reasons.append(f"クローゼットにかなり似た服があります（かぶり度 {dup_pct}%）")
    elif bm_sim >= 0.75:
        reasons.append(f"似た服がクローゼットにあります（かぶり度 {dup_pct}%）")
    else:
        reasons.append(f"クローゼットとあまり被っていません（新規性 {nov_pct}%）")

    # --- シーン一致 ---
    sc = scene_score(user_occasion, cand_tags)
    score += W_SCENE * sc

    if user_occasion:
        if sc >= 0.20:
            reasons.append(f"今回のシーン「{user_occasion}」に合いやすいです")
        else:
            reasons.append(f"今回のシーン「{user_occasion}」との一致は低めです")
    else:
        reasons.append("シーン指定がないので、シーン一致は評価していません")

    # --- 似すぎペナルティ（最優先） ---
    if bm_sim >= SAME_ITEM_TH:
        score -= SAME_ITEM_PENALTY
        reasons.append("似すぎ判定のため、大きく減点しました")

    # --- お気に入り寄せ（ただし似すぎは除外） ---
    fav_hit = (int(bm_item.get("is_favorite", 0)) == 1) and (bm_sim >= FAVORITE_SIM_TH) and (bm_sim < SAME_ITEM_TH)
    if fav_hit:
        score += FAVORITE_BONUS
        reasons.append("お気に入りに近い雰囲気なので加点しました")
    else:
        # どこが未達かを軽く説明
        if int(bm_item.get("is_favorite", 0)) != 1:
            reasons.append("お気に入り寄せ：クローゼット側のお気に入りが未設定です")
        elif bm_sim < FAVORITE_SIM_TH:
            reasons.append("お気に入り寄せ：お気に入りに十分近くありません")
        else:
            reasons.append("お気に入り寄せ：似すぎ判定のため対象外です")

    # --- 仕上げ：数値詳細は最後に1行だけ（ログ感を減らす） ---
    reasons.append(f"（詳細）おすすめ度={score:.3f} / best_sim={bm_sim:.3f}")


    return {
        "final": float(score),
        "best_sim": float(bm_sim),
        "best_item_id": int(bm_item.get("id")),
        "best_item_path": bm_item.get("image_path"),
        "reasons": reasons,
        "scene": float(sc),
        "novelty": float(novelty),
    }
