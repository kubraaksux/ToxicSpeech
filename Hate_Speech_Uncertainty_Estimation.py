"""
Enhancing Hate Speech Detection in Turkish Tweets
Using Hybrid Uncertainty Quantification

Complete pipeline: BERTurk embeddings → GMM / Mahalanobis / Entropy uncertainty → Hybrid correction

Usage:
    export HF_TOKEN="your_huggingface_token"
    export SIU_DATA_PATH="/path/to/SIU_data"
    python Hate_Speech_Uncertainty_Estimation.py
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from scipy.stats import entropy
from sklearn.decomposition import PCA, KernelPCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import MinCovDet
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, f1_score,
    accuracy_score, precision_score, recall_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from transformers import AutoTokenizer, AutoModel

# =============================================================================
# 1. Setup
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# 2. Model Loading
# =============================================================================

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
model = AutoModel.from_pretrained(
    "TR-HSD/siu-subtask2-bert-class-weight-clr-best-cv-2",
    token=os.environ.get("HF_TOKEN")
)
model = model.to(device)
model.eval()

# =============================================================================
# 3. Data Loading
# =============================================================================

HS_PATH = Path(os.environ.get("SIU_DATA_PATH", "./data/SIU_data"))

task = 'subtask2'
base_path = HS_PATH

config = {
    'subtask1': {
        'content': base_path / 'SIU-isr-pal.csv',
        'train': base_path / 'subtask1/SIU-isr-pal-traincat.csv',
        'test': base_path / 'subtask1/SIU-isr-pal-testcat.csv',
        'label': 'hs category majority',
    },
    'subtask2': {
        'content': base_path / 'SIU-refugee.csv',
        'train': base_path / 'subtask2/SIU-refugee-train.csv',
        'test': base_path / 'subtask2/SIU-refugee-test.csv',
        'label': 'hs',
    },
    'subtask3': {
        'content': base_path / 'SIU-isr-pal.csv',
        'train': base_path / 'subtask3/SIU-isr-pal-trainst.csv',
        'test': base_path / 'subtask3/SIU-isr-pal-testst.csv',
        'label': 'hs strength majority',
    },
    'subtask4': {
        'content': base_path / 'SIU-refugee.csv',
        'train': base_path / 'subtask2/SIU-refugee-train.csv',
        'test': base_path / 'subtask2/SIU-refugee-test.csv',
        'label': 'hs category',
    }
}

df_texts = pd.read_csv(config[task]['content'])
df_train_labels = pd.read_csv(config[task]['train'])
df_test_labels = pd.read_csv(config[task]['test'])

df_train = pd.concat([df_train_labels.set_index('id'), df_texts.set_index('id')], axis=1, join='inner').reset_index()
df_test = pd.concat([df_test_labels.set_index('id'), df_texts.set_index('id')], axis=1, join='inner').reset_index()

df_train = df_train.loc[:, ~df_train.columns.duplicated()].copy()
df_test = df_test.loc[:, ~df_test.columns.duplicated()].copy()

df_train['label'] = df_train[config[task]['label']].copy()
df_test['label'] = df_test[config[task]['label']].copy()

num_classes = df_train['label'].nunique()
train_labels = df_train['label'].to_numpy()
test_labels = df_test['label'].to_numpy()

print(f"Train: {len(df_train)} samples, Test: {len(df_test)} samples, Classes: {num_classes}")

# =============================================================================
# 4. Embedding Extraction
# =============================================================================

def get_embedding(text, emb_type='pooler'):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        if emb_type == 'CLS':
            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).detach().cpu().numpy()
        elif emb_type == 'mean':
            embedding = torch.mean(outputs.last_hidden_state[:, 1:-1, :], dim=1).squeeze(0).detach().cpu().numpy()
        elif emb_type == 'pooler':
            embedding = outputs.pooler_output.squeeze(0).detach().cpu().numpy()
        else:
            raise NotImplementedError(f"Unknown embedding type: {emb_type}")
    return embedding

train_texts = df_train['text'].tolist()
test_texts = df_test['text'].tolist()

train_embeddings = np.vstack([get_embedding(text, emb_type='pooler') for text in tqdm(train_texts, desc="Train embeddings")])
test_embeddings = np.vstack([get_embedding(text, emb_type='pooler') for text in tqdm(test_texts, desc="Test embeddings")])

print(f"Train embeddings: {train_embeddings.shape}, Test embeddings: {test_embeddings.shape}")

# =============================================================================
# 5. Linear Classifier (trained on embeddings)
# =============================================================================

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

classifier = LinearClassifier(input_dim=768, num_classes=num_classes).to(device)

# Train the classifier on training embeddings
train_tensor = torch.tensor(train_embeddings, dtype=torch.float32).to(device)
train_label_tensor = torch.tensor(train_labels, dtype=torch.long).to(device)

optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

classifier.train()
for epoch in range(100):
    optimizer.zero_grad()
    logits = classifier(train_tensor)
    loss = criterion(logits, train_label_tensor)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    train_logits = classifier(train_tensor)
    train_preds = torch.argmax(train_logits, dim=1).cpu().numpy()
    train_acc = accuracy_score(train_labels, train_preds)
print(f"Classifier trained — train accuracy: {train_acc:.4f}")

# Get test predictions
test_tensor = torch.tensor(test_embeddings, dtype=torch.float32).to(device)

classifier.eval()
with torch.no_grad():
    logits = classifier(test_tensor)
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    _, predicted_classes = torch.max(probabilities, dim=1)
    test_predictions = predicted_classes.cpu().numpy()

softmax_probs = probabilities.cpu().numpy()
test_acc = accuracy_score(test_labels, test_predictions)
print(f"Test accuracy: {test_acc:.4f}")

# =============================================================================
# 6. GMM Uncertainty (DDU)
# =============================================================================

scaler = StandardScaler()
train_embeddings_scaled = scaler.fit_transform(train_embeddings)
test_embeddings_scaled = scaler.transform(test_embeddings)

gmm = GaussianMixture(n_components=num_classes, covariance_type='full', random_state=42)
gmm.fit(train_embeddings_scaled)

test_uncertainties = -gmm.score_samples(test_embeddings_scaled)
incorrect_gmm_indices = np.where(test_predictions != test_labels)[0]

print(f"GMM incorrect predictions: {len(incorrect_gmm_indices)}/{len(test_labels)}")

# =============================================================================
# 7. Entropy-based Correction of GMM
# =============================================================================

entropy_threshold = 0.9
test_entropies = entropy(softmax_probs, axis=1)

incorrect_gmm_count = np.sum(test_predictions != test_labels)
entropy_corrections = 0

for idx in range(len(test_labels)):
    gmm_prediction = test_predictions[idx]
    true_class = test_labels[idx]
    entropy_prediction = 1 if test_entropies[idx] < entropy_threshold else 0

    if gmm_prediction != true_class and entropy_prediction == true_class:
        entropy_corrections += 1

correction_rate = entropy_corrections / max(incorrect_gmm_count, 1) * 100
print(f"GMM errors: {incorrect_gmm_count}, Entropy corrections: {entropy_corrections}, Rate: {correction_rate:.1f}%")

# =============================================================================
# 8. Relative Distance Estimation (RDE)
# =============================================================================

class RDESeq:
    def __init__(self, n_components=100, kernel="rbf", random_state=42):
        self.pca = KernelPCA(n_components=n_components, kernel=kernel, random_state=random_state)
        self.mcd = None
        self.is_fitted = False

    def fit(self, train_embeddings):
        X_pca = self.pca.fit_transform(train_embeddings)
        self.mcd = MinCovDet(random_state=42).fit(X_pca)
        self.is_fitted = True

    def predict(self, embeddings):
        if not self.is_fitted:
            raise Exception("RDESeq model not fitted. Call fit() first.")
        X_pca = self.pca.transform(embeddings)
        return self.mcd.mahalanobis(X_pca)

rde_estimator = RDESeq()
rde_estimator.fit(train_embeddings)
test_rd = rde_estimator.predict(test_embeddings)

print(f"RD scores range: [{test_rd.min():.2f}, {test_rd.max():.2f}]")

# =============================================================================
# 9. Entropy Correction for RD Predictions
# =============================================================================

incorrect_rd_indices = [i for i, pred in enumerate(test_predictions) if pred != test_labels[i]]
incorrect_softmax_probs = softmax_probs[incorrect_rd_indices]
incorrect_entropies = entropy(incorrect_softmax_probs, axis=1)
entropy_threshold_rd = np.percentile(incorrect_entropies, 90)

entropy_corrected_predictions = test_predictions.copy()
entropy_corrected = []

for i, entropy_value in zip(incorrect_rd_indices, incorrect_entropies):
    if entropy_value < entropy_threshold_rd:
        new_pred = 1 - test_predictions[i]
        entropy_corrected_predictions[i] = new_pred
        entropy_corrected.append(i)

print(f"RD incorrect: {len(incorrect_rd_indices)}, Entropy corrections: {len(entropy_corrected)}")

# =============================================================================
# 10. Mahalanobis Distance
# © 2023 AIRI. Licensed under MIT License.
# Adapted from: https://github.com/AIRI-Institute/hybrid_uncertainty_estimation
# =============================================================================

def compute_centroids(train_features, train_labels, class_cond=True):
    if class_cond:
        centroids = [train_features[train_labels == label].mean(axis=0)
                     for label in np.sort(np.unique(train_labels))]
        return np.asarray(centroids)
    else:
        return train_features.mean(axis=0)

def compute_covariance(centroids, train_features, train_labels, class_cond=True):
    cov = np.zeros((train_features.shape[1], train_features.shape[1]))
    if class_cond:
        for c, mu_c in enumerate(centroids):
            diff = train_features[train_labels == c] - mu_c
            cov += np.sum(diff[:, :, np.newaxis] * diff[:, np.newaxis, :], axis=0)
    else:
        diff = train_features - centroids
        cov += np.sum(diff[:, :, np.newaxis] * diff[:, np.newaxis, :], axis=0)
    cov /= train_features.shape[0]
    return np.linalg.pinv(cov)

def calculate_distance(diff, covariance):
    if diff.ndim == 1:
        diff = diff.reshape(1, -1)
    inter_result = np.matmul(np.matmul(diff, covariance), diff.T)
    return np.sqrt(np.diag(inter_result))

def mahalanobis_distance(train_features, train_labels, eval_features,
                          centroids=None, covariance=None, return_full=False):
    if centroids is None:
        centroids = compute_centroids(train_features, train_labels)
    if covariance is None:
        covariance = compute_covariance(centroids, train_features, train_labels)

    start = time.time()
    all_dists = []
    for eval_sample in eval_features:
        diff = eval_sample - centroids
        dists = calculate_distance(diff, covariance)
        all_dists.append(dists)

    all_dists = np.array(all_dists)
    end = time.time()

    if return_full:
        return all_dists, end - start
    else:
        min_dists = np.min(all_dists, axis=1)
        min_indices = np.argmin(all_dists, axis=1)
        return min_dists, end - start, min_indices

def mahalanobis_distance_marginal(train_features, train_labels, eval_features,
                                   centroids=None, covariance=None):
    centroids = centroids or compute_centroids(train_features, train_labels, class_cond=False)
    covariance = covariance or compute_covariance(centroids, train_features, train_labels, class_cond=False)

    dists = []
    for eval_sample in eval_features:
        diff = eval_sample - centroids
        dist = calculate_distance(diff, covariance)
        dists.append(dist)
    return np.array(dists)

def mahalanobis_distance_relative(train_features, train_labels, eval_features,
                                   train_centroid=None, train_covariance=None):
    if train_centroid is None or train_covariance is None:
        train_centroid = compute_centroids(train_features, train_labels)
        train_covariance = compute_covariance(train_centroid, train_features, train_labels)

    eval_centroids = compute_centroids(eval_features, np.zeros(eval_features.shape[0]))
    eval_covariance = compute_covariance(eval_centroids, eval_features, np.zeros(eval_features.shape[0]))

    md_train = mahalanobis_distance(train_features, train_labels, eval_features,
                                     centroids=train_centroid, covariance=train_covariance, return_full=True)[0]
    md_eval = mahalanobis_distance(eval_features, np.zeros(eval_features.shape[0]), eval_features,
                                    eval_centroids, eval_covariance, return_full=True)[0]

    return np.min(md_train, axis=1) - np.min(md_eval, axis=1)

# Compute distances
md_train, _, _ = mahalanobis_distance(train_embeddings, df_train['label'].values, test_embeddings, return_full=False)
md_eval, _, _ = mahalanobis_distance(test_embeddings, np.zeros(test_embeddings.shape[0]), test_embeddings, return_full=False)
relative_md = md_train - md_eval

# Full distance matrix for predictions
full_dist, _ = mahalanobis_distance(train_embeddings, df_train['label'].values, test_embeddings, return_full=True)
md_preds = np.argmin(full_dist, axis=1)

md_f1 = f1_score(df_test['label'].tolist(), md_preds, average='macro')
print(f"Mahalanobis Distance F1 (macro): {md_f1:.4f}")

# =============================================================================
# 11. Hybrid: Mahalanobis + Entropy Correction
# =============================================================================

threshold_high = np.percentile(md_train, 95)
high_uncertainty_indices = np.where(md_train > threshold_high)[0]

entropies_md = np.array([entropy(prob) for prob in softmax_probs])
entropy_threshold_md = np.percentile(entropies_md, 50)
low_entropy_indices = np.where(entropies_md < entropy_threshold_md)[0]

potential_corrections = np.intersect1d(high_uncertainty_indices, low_entropy_indices)

corrected_md_predictions = test_predictions.copy()
for idx in potential_corrections:
    corrected_md_predictions[idx] = np.argmax(softmax_probs[idx])

incorrect_before = np.sum(test_predictions != test_labels)
incorrect_after = np.sum(corrected_md_predictions != test_labels)
print(f"Hybrid correction: {incorrect_before} errors → {incorrect_after} errors ({incorrect_before - incorrect_after} fixed)")

# =============================================================================
# 12. SHAP Explainability
# =============================================================================

import shap

def classifier_predict(embeddings_np):
    """Wrapper for SHAP: takes numpy embeddings, returns probabilities."""
    t = torch.tensor(embeddings_np, dtype=torch.float32).to(device)
    classifier.eval()
    with torch.no_grad():
        logits_out = classifier(t)
        probs = torch.nn.functional.softmax(logits_out, dim=1).cpu().numpy()
    return probs

# Use KernelExplainer with a summary of training embeddings as background
background = shap.kmeans(train_embeddings, 50)
explainer = shap.KernelExplainer(classifier_predict, background)

# Explain high-uncertainty test samples
high_unc_indices = np.argsort(test_uncertainties)[-20:]
high_unc_embeddings = test_embeddings[high_unc_indices]

shap_values = explainer.shap_values(high_unc_embeddings, nsamples=100)

# Feature importance: mean absolute SHAP value per embedding dimension
if isinstance(shap_values, list):
    # Multi-class: average across classes
    mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
else:
    mean_shap = np.abs(shap_values).mean(axis=0)

top_features = np.argsort(mean_shap)[-20:][::-1]
print("\nTop 20 most important embedding dimensions (SHAP):")
for rank, feat_idx in enumerate(top_features, 1):
    print(f"  {rank:2d}. Dimension {feat_idx:3d} | mean |SHAP|: {mean_shap[feat_idx]:.6f}")

# =============================================================================
# 13. Covariate Shift Detection
# =============================================================================

shift_labels = np.hstack((np.zeros(train_embeddings.shape[0]), np.ones(test_embeddings.shape[0])))
shift_embeddings = np.vstack((train_embeddings, test_embeddings))

X_shift_train, X_shift_val, y_shift_train, y_shift_val = train_test_split(
    shift_embeddings, shift_labels, test_size=0.2, random_state=42
)

models_shift = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Support Vector Machine': SVC(probability=True)
}

print("\nCovariate Shift Detection:")
for model_name, shift_model in models_shift.items():
    shift_model.fit(X_shift_train, y_shift_train)
    y_pred_prob = shift_model.predict_proba(X_shift_val)[:, 1]
    y_pred = shift_model.predict(X_shift_val)
    auc_roc = roc_auc_score(y_shift_val, y_pred_prob)
    f1 = f1_score(y_shift_val, y_pred)
    print(f'  {model_name:30s} AUC-ROC: {auc_roc:.3f}  F1: {f1:.3f}')

# =============================================================================
# 14. Visualization (saves to figures/)
# =============================================================================

os.makedirs("figures", exist_ok=True)

# GMM errors vs entropy corrections
plt.figure(figsize=(8, 5))
plt.bar(['Incorrect GMM\nPredictions', 'Entropy\nCorrections'],
        [incorrect_gmm_count, entropy_corrections], color=['#d32f2f', '#388e3c'])
plt.title('GMM Incorrect Predictions vs. Entropy Corrections')
plt.ylabel('Number of Instances')
plt.tight_layout()
plt.savefig('figures/gmm_vs_entropy.png', dpi=150)
plt.close()

# 3D PCA
pca_3d = PCA(n_components=3)
reduced_3d = pca_3d.fit_transform(train_embeddings)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(reduced_3d[:, 0], reduced_3d[:, 1], reduced_3d[:, 2],
                     c=df_train['label'], cmap='viridis', alpha=0.6)
ax.set_title("3D PCA of Tweet Embeddings")
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.set_zlabel("PCA3")
plt.legend(*scatter.legend_elements(), title="Labels")
plt.tight_layout()
plt.savefig('figures/pca_3d.png', dpi=150)
plt.close()

# 2D PCA
pca_2d = PCA(n_components=2)
pca_result_2d = pca_2d.fit_transform(np.vstack((train_embeddings, test_embeddings)))
all_labels_viz = np.hstack((df_train['label'].tolist(), df_test['label'].tolist()))
plt.figure(figsize=(10, 8))
plt.scatter(pca_result_2d[:, 0], pca_result_2d[:, 1], c=all_labels_viz, cmap='coolwarm', alpha=0.5)
plt.colorbar(label='Class')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('2D PCA of Tweet Embeddings')
plt.tight_layout()
plt.savefig('figures/pca_2d.png', dpi=150)
plt.close()

# RD scores on PCA
X_pca = rde_estimator.pca.transform(test_embeddings)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=test_rd, cmap='viridis')
plt.colorbar(scatter, label='RD Score')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('RD Scores on PCA Components')
plt.tight_layout()
plt.savefig('figures/rd_scores_pca.png', dpi=150)
plt.close()

print("\nFigures saved to figures/")
print("Done.")
