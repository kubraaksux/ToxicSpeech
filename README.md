# Enhancing Hate Speech Detection in Turkish Tweets Using Hybrid Uncertainty Quantification

Uncertainty estimation for Turkish hate speech detection using BERTurk embeddings.

## Dataset

**SIU2023-NST** - Turkish tweets from the SIU 2023 Hate Speech Detection Contest.

## Methods

- BERTurk embeddings (768-dim pooler output)
- GMM density-based uncertainty
- Mahalanobis Distance

## Setup

```bash
pip install torch transformers scikit-learn numpy pandas matplotlib seaborn tqdm
```

## License

MIT License - Copyright (c) 2023 Kubra Aksu
