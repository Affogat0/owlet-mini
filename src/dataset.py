import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .config import IMG_SIZE, METADATA_CSV

class GazeDataset(Dataset):
    def __init__(self, csv_path=METADATA_CSV, split="train", val_ratio=0.2, transform=None):
        self.df = pd.read_csv(csv_path)
        # simple split by index
        n = len(self.df)
        split_idx = int((1 - val_ratio) * n)
        if split == "train":
            self.df = self.df.iloc[:split_idx].reset_index(drop=True)
        elif split == "val":
            self.df = self.df.iloc[split_idx:].reset_index(drop=True)
        else:
            raise ValueError("split must be 'train' or 'val'")

        # label encoding
        self.label2idx = {label: i for i, label in enumerate(sorted(self.df["label"].unique()))}
        self.idx2label = {i: l for l, i in self.label2idx.items()}
        self.df["label_idx"] = self.df["label"].map(self.label2idx)

        self.transform = transform or transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        img = self.transform(img)
        label = int(row["label_idx"])
        return img, label
