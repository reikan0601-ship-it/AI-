import json
import shutil
from pathlib import Path
import numpy as np
import torch
import open_clip
from PIL import Image

UPLOAD_DIR = Path("static/uploads")

def load_image(path: str) -> Image.Image:
    """Load image from path"""
    return Image.open(path).convert("RGB")

@torch.no_grad()
def get_embedding(model, preprocess, device, image_path: str) -> np.ndarray:
    """Get CLIP embedding for an image"""
    image = preprocess(load_image(image_path)).unsqueeze(0).to(device)
    feat = model.encode_image(image)
    feat = feat / feat.norm(dim=-1, keepdim=True)  # L2 normalization
    return feat.squeeze(0).cpu().numpy()

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity (for normalized embeddings)"""
    return float(np.dot(a, b))

def load_clip_model(model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
    """Load CLIP model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device).eval()
    return model, preprocess, device

def ensure_upload_dir():
    """Ensure upload directory exists"""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def save_image_to_uploads(src_path: Path) -> Path:
    """Save image to uploads directory with unique name"""
    ensure_upload_dir()
    
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


# タグ候補リスト（まずは少数でOK。あとで増やせる）
TAG_CANDIDATES = [
    # style
    "casual", "cute", "clean", "street", "formal",
    # season
    "spring outfit", "summer outfit", "autumn outfit", "winter outfit",
    # occasion
    "date outfit", "school outfit", "office outfit", "party outfit",
]

@torch.no_grad()
def get_tag_candidates(image_path: str, topk: int = 3):
    """
    画像に対してTAG_CANDIDATESの中から近いタグを上位topk個返す
    """
    model, preprocess, device = load_clip_model()

    image = preprocess(load_image(image_path)).unsqueeze(0).to(device)
    text = open_clip.tokenize(TAG_CANDIDATES).to(device)

    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # 類似度 -> 上位
    sims = (image_features @ text_features.T).squeeze(0)  # (num_tags,)
    top_vals, top_idx = torch.topk(sims, k=min(topk, sims.shape[0]))

    results = []
    for v, i in zip(top_vals.tolist(), top_idx.tolist()):
        results.append({"tag": TAG_CANDIDATES[i], "score": float(v)})

    return results
