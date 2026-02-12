# app/logic/embeddings.py
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import open_clip
from PIL import Image


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


@lru_cache(maxsize=1)
def get_clip_bundle():
    """
    Flaskプロセス内で1回だけモデルをロードして使い回す
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "ViT-B-32"
    pretrained = "laion2b_s34b_b79k"

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device).eval()
    return model, preprocess, device


@torch.no_grad()
def get_image_embedding(image_path: str) -> np.ndarray:
    model, preprocess, device = get_clip_bundle()
    image = preprocess(load_image(image_path)).unsqueeze(0).to(device)
    feat = model.encode_image(image)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).float().cpu().numpy()
