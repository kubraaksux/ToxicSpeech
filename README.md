# Enhancing Hate Speech Detection in Turkish Tweets Using Hybrid Uncertainty Quantification

A hybrid uncertainty estimation system for Turkish hate speech detection that combines GMM density estimation, Mahalanobis distance, and entropy-based methods to identify uncertain predictions — enabling reliable content moderation with human-in-the-loop review.

## Architecture

```
Turkish Tweet
     │
     ▼
┌─────────────────────────┐
│  BERTurk Tokenizer +    │
│  Embedding (768-dim)    │
│  pooler_output          │
└────────────┬────────────┘
             │
     ┌───────┼───────┐
     │       │       │
     ▼       ▼       ▼
  ┌─────┐ ┌──────┐ ┌─────────┐
  │ GMM │ │ Maha │ │Softmax  │
  │Score│ │lanob.│ │Entropy  │
  └──┬──┘ └──┬───┘ └────┬────┘
     │       │          │
     ▼       ▼          ▼
┌──────────────────────────────┐
│   Hybrid Decision Engine     │
│                              │
│  All agree     → Auto-label  │
│  High uncert.  → Human review│
│  GMM wrong +   → Entropy     │
│  low entropy     correction  │
└──────────────────────────────┘
```

## Dataset

**SIU2023-NST** — 5,000 Turkish tweets from the Signal Processing and Communications Applications Conference 2023 Hate Speech Detection Contest.

| Subtask | Data | Task | Label |
|---------|------|------|-------|
| 1 | SIU-isr-pal.csv | HS Category (multi-class) | `hs category majority` |
| **2 (primary)** | **SIU-refugee.csv** | **Binary HS detection** | **`hs`** |
| 3 | SIU-isr-pal.csv | HS Strength | `hs strength majority` |
| 4 | SIU-refugee.csv | HS Category | `hs category` |

Organized by: Hrant Dink Foundation, Sabanci University, Bogazici University (EU-funded).

## Methods

### Embedding
- **Model**: `dbmdz/bert-base-turkish-cased` (BERTurk, 110M params)
- **Fine-tuned**: `TR-HSD/siu-subtask2-bert-class-weight-clr-best-cv-2`
- **Output**: 768-dimensional pooler embeddings

### Uncertainty Estimation (3 methods)

| Method | Type | What It Captures |
|--------|------|-----------------|
| **GMM** | Density-based | Is this sample in a dense or sparse region of embedding space? |
| **Mahalanobis Distance** | Distance-based | How far is this from class centroids (covariance-aware)? |
| **Entropy** | Confidence-based | Is the classifier spread across classes? |

### Hybrid Correction
GMM predictions with high error rate are corrected using entropy thresholding — **~82% of GMM errors recovered**.

### Additional Analysis
- **Relative Mahalanobis Distance**: Class-conditional MD minus marginal MD for normalized uncertainty
- **Robust Distance Estimation (RDE)**: Kernel PCA + Minimum Covariance Determinant
- **Covariate Shift Detection**: Train vs. test distribution comparison using LR, KNN, DT, SVM classifiers
- **SHAP Explainability**: KernelExplainer on embedding dimensions for feature-level attribution of uncertain predictions
- **Visualization**: 3D PCA, 2D PCA, t-SNE with misclassification overlay

## Sample Output

```
Classification Results:
  Accuracy:  0.7360
  F1 Score:  0.7145
  Precision: 0.7280
  Recall:    0.7015
  ROC AUC:   0.8012

Uncertainty Estimation:
  GMM flagged:          312 / 1000 test samples as uncertain
  Mahalanobis flagged:  287 / 1000 test samples as uncertain
  Entropy flagged:      245 / 1000 test samples as uncertain

Hybrid Correction:
  GMM errors corrected by entropy: ~82%
  Final accuracy after correction: 0.7840
```

> **Note:** Exact numbers depend on train/test split randomness and model checkpoint.

## Results

Figures are generated when you run the notebook or script. They are saved to the `figures/` directory:

- `figures/gmm_vs_entropy.png` — GMM incorrect predictions vs. entropy corrections
- `figures/pca_3d.png` — 3D PCA of tweet embeddings (class distribution)
- `figures/pca_2d.png` — 2D PCA of embeddings
- `figures/rd_scores_pca.png` — RD scores on PCA components

## Project Structure

```
ToxicSpeech/
├── Hate_Speech_Uncertainty_Estimation.ipynb  # Interactive notebook (full pipeline + visualizations)
├── Hate_Speech_Uncertainty_Estimation.py     # Runnable script (same pipeline, saves figures)
├── requirements.txt
├── README.md
├── LICENSE
└── figures/                                  # Generated when you run the code
```

- **Notebook** (`.ipynb`): Interactive exploration with inline plots, step-by-step analysis, Venn diagrams, calibration curves
- **Script** (`.py`): End-to-end pipeline, prints results to console, saves figures to `figures/`

## Setup

```bash
# Clone the repository
git clone https://github.com/kubraaksux/ToxicSpeech.git
cd ToxicSpeech

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export HF_TOKEN="your_huggingface_token"
export SIU_DATA_PATH="/path/to/SIU_data"

# Run (choose one)
jupyter notebook Hate_Speech_Uncertainty_Estimation.ipynb
python Hate_Speech_Uncertainty_Estimation.py
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- scikit-learn
- NumPy, Pandas, SciPy
- Matplotlib, Seaborn
- matplotlib-venn
- SHAP
- tqdm

## References

- AIRI Hybrid Uncertainty Estimation: [GitHub](https://github.com/AIRI-Institute/hybrid_uncertainty_estimation)
- BERTurk: [HuggingFace](https://huggingface.co/dbmdz/bert-base-turkish-cased)
- SIU2023-NST Shared Task: [VPALab](https://www.vpalab.com/events/siu2023-nst)
- Miok et al. (2021) "To BAN or Not to BAN: Bayesian Attention Networks for Reliable Hate Speech Detection"
- Lee et al. (2018) "A Simple Unified Framework for Detecting Out-of-Distribution Samples" (NeurIPS)

## License

MIT License — Copyright (c) 2023 Kubra Aksu
