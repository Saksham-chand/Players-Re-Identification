import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from tqdm import tqdm

# Setup transforms
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((128, 64)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

# Feature extractor
class Embedder(nn.Module):
    def __init__(self):
        super().__init__()
        model = resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(model.children())[:-1])  # Remove FC layer

    def forward(self, x):
        with torch.no_grad():
            feats = self.backbone(x)
            return feats.view(x.size(0), -1)

def load_and_embed_images(folder, model):
    embeddings = {}
    for img_name in tqdm(os.listdir(folder), desc=f"Embedding {folder}"):
        path = os.path.join(folder, img_name)
        img = cv2.imread(path)
        if img is None:
            continue
        img = transform(img).unsqueeze(0)
        embedding = model(img).squeeze().numpy()
        embeddings[img_name] = embedding
    return embeddings

def match_players(broadcast_embeds, tacticam_embeds):
    matches = {}
    for tac_name, tac_vec in tacticam_embeds.items():
        best_match = None
        best_score = -1
        for broad_name, broad_vec in broadcast_embeds.items():
            score = cosine_similarity([tac_vec], [broad_vec])[0][0]
            if score > best_score:
                best_score = score
                best_match = broad_name
        matches[tac_name] = (best_match, best_score)
    return matches

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--broadcast', required=True, help='Folder with broadcast crops')
    parser.add_argument('--tacticam', required=True, help='Folder with tacticam crops')
    args = parser.parse_args()

    model = Embedder().eval()

    # Step 1: Load and embed all images
    broadcast_embeds = load_and_embed_images(args.broadcast, model)
    tacticam_embeds = load_and_embed_images(args.tacticam, model)

    # Step 2: Match
    matches = match_players(broadcast_embeds, tacticam_embeds)

    # Step 3: Print results
    print("\nPlayer Matching Results:")
    for tac, (broad, score) in matches.items():
        print(f"Tacticam player {tac} → Broadcast player {broad} (similarity: {score:.4f})")
