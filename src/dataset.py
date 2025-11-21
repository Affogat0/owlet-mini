# src/dataset.py

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .config import IMG_SIZE, METADATA_CSV


class GazeDataset(Dataset):
    def __init__(
        self,
        csv_path: str = METADATA_CSV,
        split: str = "train",
        val_ratio: float = 0.2,
        transform=None,
    ):
        """
        csv_path: path to metadata.csv with columns: image_path, label, subject_id
        split: 'train' or 'val'
        val_ratio: fraction of data to reserve for validation
        """

        # Load full metadata
        full_df = pd.read_csv(csv_path)

        # Build a single label mapping shared by train & val
        self.label2idx = {
            label: i for i, label in enumerate(sorted(full_df["label"].unique()))
        }
        self.idx2label = {i: l for l, i in self.label2idx.items()}

        full_df["label_idx"] = full_df["label"].map(self.label2idx)

        # Simple train/val split by index
        n = len(full_df)
        split_idx = int((1.0 - val_ratio) * n)

        if split == "train":
            self.df = full_df.iloc[:split_idx].reset_index(drop=True)
        elif split == "val":
            self.df = full_df.iloc[split_idx:].reset_index(drop=True)
        else:
            raise ValueError("split must be 'train' or 'val'")

        # Default transform: resize → tensor → normalize
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # image_path in CSV should be relative to project root
        img = Image.open(row["image_path"]).convert("RGB")
        img = self.transform(img)

        label = int(row["label_idx"])
        return img, label
