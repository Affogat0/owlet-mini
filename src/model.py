import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights

from .config import NUM_CLASSES

class TinyViTGaze(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True):
        super().__init__()
        # Start from a ViT backbone; you can switch to smaller if needed
        if pretrained:
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            self.backbone = vit_b_16(weights=weights)
        else:
            self.backbone = vit_b_16(weights=None)

        # Replace head
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                # Keep head trainable
                if "heads.head" not in name:
                    param.requires_grad = False

    def forward(self, x):
        return self.backbone(x)

def build_model(device="cuda"):
    model = TinyViTGaze()
    return model.to(device)

