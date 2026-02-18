import os
import torch
from PIL import Image
import clip
import numpy as np

def quick_demo():
    print("--- Step 1: Environment Check ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device used: {device}")
    if device == "cuda":
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    
    # 1. Load Model
    print("\n--- Step 2: Loading CLIP Model (ViT-B/32) ---")
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # 2. Prepare Data
    print("\n--- Step 3: Preparing Test Data ---")
    image_path = "CLIP.png" 
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    # The diagram shows CLIP architecture, let's see if CLIP knows itself!
    text_descriptions = ["a diagram of CLIP architecture", "a photo of a cat", "a picture of a dog"]
    text = clip.tokenize(text_descriptions).to(device)

    # 3. Inference
    print("\n--- Step 4: Inference (Encoding & Matching) ---")
    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # 4. Show Results
    print("\n" + "="*30)
    print("       FINAL RESULTS")
    print("="*30)
    for i, desc in enumerate(text_descriptions):
        print(f"Description '{desc}': {probs[0][i]*100:6.2f}%", flush=True)
    
    print("-" * 30)
    best_match = text_descriptions[np.argmax(probs)]
    print(f"WINNER: 【{best_match}】", flush=True)
    print("="*30 + "\n")

if __name__ == "__main__":
    quick_demo()
