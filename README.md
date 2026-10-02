# CNN for Malaria Cell Detection and Feature Extraction

This project trains a convolutional neural network (CNN) to classify malaria cell images as either:

- Parasitized
- Uninfected

It also exposes a feature extractor that can be used to obtain learned deep representations from the trained model for downstream tasks like clustering, anomaly detection, or transfer learning.

## Dataset structure

Organize your dataset like this:

```text
data/
├── train/
│   ├── Parasitized/
│   │   ├── img_001.png
│   │   └── ...
│   └── Uninfected/
│       ├── img_001.png
│       └── ...
├── val/
│   ├── Parasitized/
│   └── Uninfected/
└── test/
    ├── Parasitized/
    └── Uninfected/
```

If you are using the NIH Malaria dataset, it commonly follows a similar pattern with `Parasitized` and `Uninfected` subfolders.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
python src/train_cnn.py \
  --train-dir data/train \
  --val-dir data/val \
  --epochs 20 \
  --batch-size 32 \
  --output-dir checkpoints
```

This will train the CNN and save the best model as:

```text
checkpoints/best_model.keras
```

## Feature extraction

```bash
python src/extract_features.py \
  --model-path checkpoints/best_model.keras \
  --data-dir data/val \
  --output-file features/val_features.npz
```

This saves:

- `features`: learned feature vectors from the CNN's `features` layer
- `labels`: numeric class labels
- `class_indices`: mapping from class names to indices

## Project files

- `src/train_cnn.py` — model creation and training pipeline
- `src/extract_features.py` — feature extraction from trained CNN
- `requirements.txt` — Python dependencies

## Notes

- The model uses a CNN with a `GlobalAveragePooling2D` layer named `features` so the feature vectors are easy to extract.
- The training pipeline is designed for binary classification (`Parasitized` vs `Uninfected`).
- You can replace the dataset directories with your own images and keep the same structure.
