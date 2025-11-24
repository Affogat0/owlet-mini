# owlet-mini

## Results (NIT Karnataka subset)

Trained a ViT-B/16 (ImageNet-pretrained, frozen backbone, learned classifier head) on
a 3-class subset of the NIT Karnataka eye-tracking dataset (left / center / right).

| Epochs | Train Accuracy | Val Accuracy |
|--------|----------------|--------------|
| 3      | ~0.88          | ~0.38        |

After going through more epochs, the accuracy is continuing to rise.

| Epochs | Train Accuracy | Val Accuracy | Best Epoch |
|--------|----------------|--------------|------------|
| 15     | ~0.97          | ~0.90        | 15         |

These results use a simple train/val split and minimal hyperparameter tuning, and are
intended as a proof of concept for the end to end pipeline (data → ViT → training →
checkpoint → inference).


### Dataset Citation

This project uses the NIT Karnataka "eye_tracker_data" dataset:

Bhat, Shravan; Upadhyaya, Skanda; Rao, Siddhanth; Chemmangat, Krishnan (2023).  
*eye_tracker_data*. Mendeley Data, V1. https://doi.org/10.17632/vy4n28334m.1
