import os
import numpy as np
import torch
import open_clip
from PIL import Image

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

@torch.no_grad()
def get_embedding(model, preprocess, device, image_path: str) -> np.ndarray:
    image = preprocess(load_image(image_path)).unsqueeze(0).to(device)
    feat = model.encode_image(image)
    feat = feat / feat.norm(dim=-1, keepdim=True)  # L2正規化（超重要）
    return feat.squeeze(0).cpu().numpy()

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # 正規化済みなのでdotでOK

def main():
    img1 = "app/static/uploads/a.jpg"
    img2 = "app/static/uploads/b.jpg"
    img3 = "app/static/uploads/c.jpg" 

    print("cwd:", os.getcwd())
    print("img1 exists:", os.path.exists(img1))
    print("img2 exists:", os.path.exists(img2))
    print("img3 exists:", os.path.exists(img3))
   
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    model_name = "ViT-B-32"
    pretrained = "laion2b_s34b_b79k"

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device).eval()

    emb1 = get_embedding(model, preprocess, device, img1)
    emb2 = get_embedding(model, preprocess, device, img2)
    emb3 = get_embedding(model, preprocess, device, img3)

    print("embedding shape:", emb1.shape)  # 例: (512,)
    sim12 = cosine_sim(emb1, emb2)
    sim13 = cosine_sim(emb1, emb3)
    sim23 = cosine_sim(emb2, emb3)

    print(f"sim(a,b): {sim12:.4f}")
    print(f"sim(a,c): {sim13:.4f}")
    print(f"sim(b,c): {sim23:.4f}")

if __name__ == "__main__":
    main()
