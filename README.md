# Malaria Cell CNN Pipeline

This project includes a complete end-to-end CNN pipeline for malaria classification.

## Dataset layout

Use this structure:

```text
data/
├── train/
│   ├── Parasitized/
│   └── Uninfected/
├── val/
│   ├── Parasitized/
│   └── Uninfected/
└── test/
    ├── Parasitized/
    └── Uninfected/
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Train and evaluate

```bash
python src/train_cnn.py \
  --train-dir data/train \
  --val-dir data/val \
  --test-dir data/test \
  --epochs 25 \
  --batch-size 32 \
  --image-size 224 \
  --output-dir checkpoints
```

## Outputs

The script saves:

- `checkpoints/best_model.keras` — best trained model
- `checkpoints/metrics.json` — final accuracy, precision, recall, and F1
- `checkpoints/classification_report.txt` — full precision/recall/F1 report
- `checkpoints/confusion_matrix.png` — confusion matrix heatmap
- `checkpoints/training_history.json` — loss and metric history

## Included pipeline features

- data augmentation for improved generalization
- validation and test evaluation
- confusion matrix visualization
- precision/recall/F1 metrics
- model checkpointing and early stopping
- feature extraction support via the existing `src/extract_features.py` script
