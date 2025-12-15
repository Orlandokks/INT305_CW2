# INT305 — Coursework 2: CIFAR-10 Image Classification with CNNs (PyTorch)

This repository contains my **INT305 Coursework 2** implementation and analysis for **single-label image classification** on **CIFAR-10** using **Convolutional Neural Networks (CNNs)** in **PyTorch**.

The project is structured as a mini end-to-end pipeline:
- a **baseline CNN** (small/shallow) to establish a strong starting point,
- an **ImprovedNet** (deeper + regularised + stronger augmentation + modern optimisation),
- **training / validation checkpointing**, inference, and **multi-metric evaluation** (accuracy, per-class precision/recall/F1, confusion matrix, confidence-based qualitative analysis).

---

## Key Results (CIFAR-10 Test Set)

| Model | Test Loss | Top-1 Accuracy | Macro-F1 |
|------|-----------:|---------------:|---------:|
| **BaselineNet** | ~0.91 | **~68.5%** | ~69% |
| **ImprovedNet** | ~0.29 | **~91.4%** | ~91% |

Beyond the headline accuracy jump, the improved model shows:
- **more balanced per-class performance** (especially on harder animal classes),
- **cleaner confusion matrix** (errors concentrate on genuinely ambiguous pairs),
- **fewer extremely over-confident wrong predictions** (still not perfect calibration, but improved).

---

## What I Built

### 1) BaselineNet (reference CNN)
A compact CNN to establish a reproducible baseline and expose failure modes.

**Training setup highlights**
- Standard cross-entropy objective
- SGD optimisation
- Train/val monitoring + **best-checkpoint saving based on validation loss**

**Diagnostics performed**
- learning curves (train/val loss + accuracy)
- per-class precision/recall/F1 + macro-F1
- confusion matrix
- **top confident correct vs top confident wrong** predictions (with softmax confidence)

This baseline is intentionally simple, which makes it great for understanding:
- where capacity is insufficient (fine-grained classes),
- where inductive bias is weak (shape vs texture),
- and how **confidence can be misleading** (over-confident mistakes).

---

### 2) ImprovedNet (my proposed method)
A deeper and more robust CNN designed from baseline diagnostics.

**Architecture**
- 3 convolutional stages, each roughly following:
  - `Conv(3×3) → BatchNorm → ReLU → Conv(3×3) → BatchNorm → ReLU → MaxPool(2×2)`
- Channel progression: **3 → 32 → 64 → 128**
- Classifier head:
  - `Flatten(128×4×4=2048) → Dropout(0.5) → Linear(2048→256) → ReLU → Dropout(0.5) → Linear(256→10)`

**Data augmentation (stronger + label-preserving)**
- RandomCrop + HorizontalFlip
- RandomRotation
- ColorJitter
- RandomErasing
- Normalisation with CIFAR-10 statistics

**Optimisation upgrades**
- Adam + weight decay
- Cosine annealing learning-rate schedule
- Kaiming initialisation
- Same validation-based checkpointing strategy

**Why it works**
- deeper hierarchy + larger effective receptive field → better feature abstraction
- BatchNorm stabilises training at depth
- Dropout + weight decay reduce overfitting and improve robustness
- stronger augmentation improves invariance to nuisance factors (pose/background/occlusion)

---

## Inference Pipeline (Single Image → Prediction)
The inference path is kept explicit and reproducible:

1. **Preprocess**: tensor conversion + normalisation  
2. **Forward pass** (`model.eval()`): get logits `z ∈ R^10`  
3. **Softmax**: convert logits to probabilities  
4. **Decision**: `argmax(probabilities)`  
5. **Confidence**: `max(probabilities)` used for qualitative ranking

---

## Evaluation: More Than Accuracy
To understand a multi-class classifier, I report and analyse:

- **Overall accuracy**
- **Per-class precision / recall / F1**
- **Macro-F1** (class-balanced view)
- **Confusion matrix** (structured error patterns)
- **Confidence-based analysis**:
  - top confident correct predictions (what the model “really understands”)
  - top confident wrong predictions (failure modes + calibration issues)

---
