# owlet-mini

A proof-of-concept eye-gaze classifier: fine-tunes a Vision Transformer
(ViT-B/16, ImageNet-pretrained, frozen backbone + learned classifier head)
to predict gaze direction (left / center / right) from webcam-style eye
images.

This is a smaller, from-scratch companion to a CNN-based eye-movement
research poster I presented — built to explore how a pretrained ViT
backbone performs on the same kind of task with minimal fine-tuning.

## Pipeline

```
Raw images (NIT Karnataka dataset) 
            ↓ 
Preprocessing / labeling 
            ↓ 
ViT-B/16 (frozen backbone, trainable classifier head) 
            ↓ 
Training loop (checkpointing) 
            ↓ 
Inference on held-out val split
```

## Setup

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Running it

    # Train
    python3 -m src.train

    # Run inference on a single image
    python3 -m src.infer_image --image <path_to_image> --checkpoint <path_to_checkpoint>

Notebooks in `notebooks/` walk through data exploration and result
visualization if you'd rather step through it interactively.

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

The jump from 3 → 15 epochs (val accuracy 0.38 → 0.90) came from
letting the classifier head converge further on a frozen backbone. No data augmentation, regularization, or LR scheduling was used, just additional training epochs.
Given the small validation set, treat the 0.90 figure as a proof of concept signal rather than a benchmark result.

These results use a simple train/val split and minimal hyperparameter tuning, and are
intended as a proof of concept for the end to end pipeline (data → ViT → training →
checkpoint → inference).

## What this is (and isn't)

This validates that a frozen pretrained ViT backbone can learn a
3-class gaze direction task from a small dataset with just a fine tuned
classifier head. It doesn't yet handle continuous gaze estimation,
multiple subjects/webcam angles, or a proper train/val/test split with
cross-validation. Natural next steps: expand beyond 3 discrete classes,
validate across subjects not seen in training, and compare against a
non-frozen fine-tuning baseline.

### Dataset Citation

This project uses the NIT Karnataka "eye_tracker_data" dataset:

Bhat, Shravan; Upadhyaya, Skanda; Rao, Siddhanth; Chemmangat, Krishnan (2023).  
*eye_tracker_data*. Mendeley Data, V1. https://doi.org/10.17632/vy4n28334m.1
