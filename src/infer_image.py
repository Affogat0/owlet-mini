import argparse
import torch
from PIL import Image
from torchvision import transforms

from .config import IMG_SIZE, DEVICE, CHECKPOINT_DIR
from .model import build_model
from .dataset import GazeDataset  # to reuse label mapping

def load_label_mapping():
    # instantiate dataset to reuse its mapping
    ds = GazeDataset(split="train")
    return ds.idx2label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str,
                        default=f"{CHECKPOINT_DIR}/best_model.pt")
    args = parser.parse_args()

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    model = build_model(device=device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    idx2label = load_label_mapping()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    img = Image.open(args.image).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        pred_idx = logits.argmax(dim=1).item()
        pred_label = idx2label[pred_idx]

    print(f"Predicted gaze region: {pred_label}")

if __name__ == "__main__":
    main()
