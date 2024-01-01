# Load model directly
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
model = AutoModel.from_pretrained("TR-HSD/siu-subtask2-bert-class-weight-clr-best-cv-2", use_auth_token='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')

from pathlib import Path
import pandas as pd

HS_PATH = Path('/Users/kub/Desktop/FINAL/SIU_data')

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

# Load dataframes from the local files
df_texts = pd.read_csv(config[task]['content'])
df_train_labels = pd.read_csv(config[task]['train'])
df_test_labels = pd.read_csv(config[task]['test'])

# Combine text with labels using the 'id' column to join on
df_train = pd.concat([df_train_labels.set_index('id'), df_texts.set_index('id')], axis=1, join='inner').reset_index()
df_test = pd.concat([df_test_labels.set_index('id'), df_texts.set_index('id')], axis=1, join='inner').reset_index()

# Remove duplicate columns if any
df_train = df_train.loc[:, ~df_train.columns.duplicated()].copy()
df_test = df_test.loc[:, ~df_test.columns.duplicated()].copy()

# Set the label column
df_train['label'] = df_train[config[task]['label']].copy()
df_test['label'] = df_test[config[task]['label']].copy()

# Calculate the number of labels
config[task]['num_labels'] = df_train['label'].value_counts().shape[0]


import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_embedding(text, emb_type='CLS'):
      inputs = tokenizer(text, return_tensors="pt")
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
            raise NotImplementedError
      return embedding


import numpy as np
from tqdm import tqdm
train_texts = df_train['text'].tolist()
test_texts = df_test['text'].tolist()
train_embeddings = np.vstack([get_embedding(text, emb_type='pooler') for text in tqdm(train_texts)])
test_embeddings = np.vstack([get_embedding(text, emb_type='pooler') for text in tqdm(test_texts)])

# © 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research Institute" (AIRI). All rights reserved.
# Licensed under the MIT License
# Code taken and refactored from https://github.com/AIRI-Institute/hybrid_uncertainty_estimation/blob/master/src/ue4nlp/mahalanobis_distance.py

# Gokce, burasi senin eski kodun mahalanobis icin:

""" from tqdm import tqdm
import numpy as np
import time
import logging

log = logging.getLogger()

def compute_centroids(train_features, train_labels, class_cond=True): """
    """
    Computes the centroids of the given training features.

    Parameters:
    - train_features (ndarray): The features of the training data.
    - train_labels (ndarray): The labels of the training data.
    - class_cond (bool): Whether to compute class-conditional centroids.

    Returns:
    - ndarray: The computed centroids.
    """
"""
    if class_cond:
        centroids = [
            train_features[train_labels == label].mean(axis=0)
            for label in np.sort(np.unique(train_labels))
        ]
        return np.asarray(centroids)
    else:
        return train_features.mean(axis=0)

def compute_covariance(centroids, train_features, train_labels, class_cond=True):
    """
"""
    Computes the covariance matrix of the given training features.

    Parameters:
    - centroids (ndarray): The computed centroids.
    - train_features (ndarray): The features of the training data.
    - train_labels (ndarray): The labels of the training data.
    - class_cond (bool): Whether to compute class-conditional covariance.

    Returns:
    - ndarray: The inverse of the computed covariance matrix.
    """
"""
    cov = np.zeros((train_features.shape[1], train_features.shape[1]))
    if class_cond:
        for c, mu_c in tqdm(enumerate(centroids)):
            cov += sum((x - mu_c)[:, None] @ (x - mu_c)[None, :] for x in train_features[train_labels == c])
    else:
        cov += sum((x - centroids)[:, None] @ (x - centroids)[None, :] for x in train_features)
    cov /= train_features.shape[0]
    return np.linalg.pinv(cov)  # Using the Moore-Penrose inverse as it's more robust



def calculate_distance(diff, covariance):
    """
"""
    Calculates Mahalanobis distance given the difference and covariance matrices.

    Parameters:
    - diff (numpy.ndarray): Difference matrix
    - covariance (numpy.ndarray): Covariance matrix

    Returns:
    - numpy.ndarray: Mahalanobis distance matrix
    """
"""
    inter_result = np.matmul(np.matmul(diff, covariance), diff.transpose(0, 2, 1))
    return np.asarray([np.diag(result) for result in inter_result])

def mahalanobis_distance(
    train_features,
    train_labels,
    eval_features,
    centroids=None,
    covariance=None,
    return_full=False,
):
    """
"""
    Computes Mahalanobis distance between evaluation features and training data centroids.

    Parameters:
    - train_features (numpy.ndarray): Training feature matrix
    - train_labels (numpy.ndarray): Training label vector
    - eval_features (numpy.ndarray): Evaluation feature matrix
    - centroids (numpy.ndarray, optional): Centroids of the training data classes
    - covariance (numpy.ndarray, optional): Covariance matrix of the training data
    - return_full (bool, optional): If True, returns the full distance matrix, otherwise returns the minimum distance

    Returns:
    - tuple: A tuple containing the distance matrix or minimum distances, and the computation time
    """
"""
    centroids = centroids or compute_centroids(train_features, train_labels)
    covariance = covariance or compute_covariance(centroids, train_features, train_labels)

    diff = eval_features[:, None, :] - centroids[None, :, :]
    print(eval_features.shape, diff.shape)
    start = time.time()
    print(covariance.shape)
    dists = calculate_distance(diff, covariance)
    end = time.time()

    if return_full:
        return dists, end - start
    else:
        return np.min(dists, axis=1), end - start, np.argmin(dists, axis=1)

def mahalanobis_distance_marginal(
    train_features,
    train_labels,
    eval_features,
    centroids=None,
    covariance=None
):
    """
"""
    Computes the marginal Mahalanobis distances for the evaluation features.

    Parameters:
    - train_features (numpy.ndarray): Training feature matrix
    - train_labels (numpy.ndarray): Training label vector
    - eval_features (numpy.ndarray): Evaluation feature matrix
    - centroids (numpy.ndarray, optional): Centroids of the training data classes
    - covariance (numpy.ndarray, optional): Covariance matrix of the training data

    Returns:
    - numpy.ndarray: Vector of marginal Mahalanobis distances
    """
"""
    centroids = centroids or compute_centroids(train_features, train_labels, class_cond=False)
    covariance = covariance or compute_covariance(centroids, train_features, train_labels, class_cond=False)

    diff = eval_features - centroids[None, :]
    dists = np.matmul(np.matmul(diff, covariance), diff.T)
    return np.diag(dists)


def mahalanobis_distance_relative(
    train_features,
    train_labels,
    eval_features,
    centroids=None,
    covariance=None,
    train_centroid=None,
    train_covariance=None,
):
    """
"""
    Computes the relative Mahalanobis distances for the evaluation features.

    Parameters:
    - train_features (numpy.ndarray): Training feature matrix
    - train_labels (numpy.ndarray): Training label vector
    - eval_features (numpy.ndarray): Evaluation feature matrix
    - centroids (numpy.ndarray, optional): Centroids of the training data classes
    - covariance (numpy.ndarray, optional): Covariance matrix of the training data
    - train_centroid (numpy.ndarray, optional): Global centroid of the training data
    - train_covariance (numpy.ndarray, optional): Global covariance matrix of the training data

    Returns:
    - numpy.ndarray: Vector of relative Mahalanobis distances
    """
"""
    centroids = centroids or compute_centroids(train_features, train_labels)
    covariance = covariance or compute_covariance(centroids, train_features, train_labels)

    diff = eval_features[:, None, :] - centroids[None, :, :]
    dists = calculate_distance(diff, covariance)

    md_marginal = mahalanobis_distance_marginal(
        train_features, train_labels, eval_features, train_centroid, train_covariance
    )
    return np.min(dists - md_marginal[:, None], axis=1) 
    """

# © 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research Institute" (AIRI). All rights reserved.
# Licensed under the MIT License
# Code taken and refactored from https://github.com/AIRI-Institute/hybrid_uncertainty_estimation/blob/master/src/ue4nlp/mahalanobis_distance.py

from tqdm import tqdm
import numpy as np
import time
import logging

log = logging.getLogger()

def compute_centroids(train_features, train_labels, class_cond=True):
    """
    Computes the centroids of the given training features.
    This function remains unchanged as it correctly calculates centroids for given features.
    """
    if class_cond:
        centroids = [train_features[train_labels == label].mean(axis=0) for label in np.sort(np.unique(train_labels))]
        return np.asarray(centroids)
    else:
        return train_features.mean(axis=0)

def compute_covariance(centroids, train_features, train_labels, class_cond=True):
    """
    Computes the covariance matrix of the given training features.
    This function remains unchanged as it correctly computes covariance matrices.
    """
    cov = np.zeros((train_features.shape[1], train_features.shape[1]))
    if class_cond:
        for c, mu_c in tqdm(enumerate(centroids)):
            cov += np.sum((train_features[train_labels == c] - mu_c)[:, :, np.newaxis] * (train_features[train_labels == c] - mu_c)[:, np.newaxis, :], axis=0)
    else:
        cov += np.sum((train_features - centroids)[:, :, np.newaxis] * (train_features - centroids)[:, np.newaxis, :], axis=0)
    cov /= train_features.shape[0]
    return np.linalg.pinv(cov)

def calculate_distance(diff, covariance):
    """
    Calculates Mahalanobis distance given the difference and covariance matrices.
    Updated to handle one-dimensional 'diff' inputs by reshaping them. This ensures compatibility 
    with single-sample inputs, addressing a previously identified broadcasting issue.
    """
    if diff.ndim == 1:
        diff = diff.reshape(1, -1)
    inter_result = np.matmul(np.matmul(diff, covariance), diff.T)
    return np.sqrt(np.diag(inter_result))

def mahalanobis_distance(train_features, train_labels, eval_features, centroids=None, covariance=None, return_full=False):
    """
    Computes Mahalanobis distance between evaluation features and training data centroids.
    Updated to process each evaluation feature separately, avoiding issues with broadcasting and matrix dimensions.
    This allows for more flexible and error-free computation of distances.
    """
    centroids = centroids or compute_centroids(train_features, train_labels)
    covariance = covariance or compute_covariance(centroids, train_features, train_labels)

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

def mahalanobis_distance_marginal(train_features, train_labels, eval_features, centroids=None, covariance=None):
    """
    Computes the marginal Mahalanobis distances for the evaluation features.
    Updated to loop over individual evaluation samples, addressing the broadcasting issue and ensuring consistency in calculations.
    """
    centroids = centroids or compute_centroids(train_features, train_labels, class_cond=False)
    covariance = covariance or compute_covariance(centroids, train_features, train_labels, class_cond=False)

    dists = []
    for eval_sample in eval_features:
        diff = eval_sample - centroids
        dist = calculate_distance(diff, covariance)
        dists.append(dist)
    
    return np.array(dists)

def mahalanobis_distance_relative(train_features, train_labels, eval_features, centroids=None, covariance=None, train_centroid=None, train_covariance=None):
    """
    Computes the relative Mahalanobis distances for the evaluation features.
    Utilizes the updated functions above to calculate relative distances. This approach aligns 
    with the new methodology of handling individual samples, thus ensuring compatibility and correctness.
    """
    train_centroids = centroids or compute_centroids(train_features, train_labels)
    train_covariance = covariance or compute_covariance(train_centroids, train_features, train_labels)

    eval_centroids = compute_centroids(eval_features, np.zeros(eval_features.shape[0]))  # Assuming zero labels for eval set
    eval_covariance = compute_covariance(eval_centroids, eval_features, np.zeros(eval_features.shape[0]))

    md_train = mahalanobis_distance(train_features, train_labels, eval_features, train_centroids, train_covariance, return_full=True)
    md_eval = mahalanobis_distance(eval_features, np.zeros(eval_features.shape[0]), eval_features, eval_centroids, eval_covariance, return_full=True)

    return np.min(md_train, axis=1) - np.min(md_eval, axis=1)


# Compute the Mahalanobis distance for the training and test (evaluation) data
md_train, _, _ = mahalanobis_distance(train_embeddings, df_train['label'].values, test_embeddings, return_full=False)
md_eval, _, _ = mahalanobis_distance(test_embeddings, np.zeros(test_embeddings.shape[0]), test_embeddings, return_full=False)

# Calculate relative Mahalanobis distance
relative_md = md_train - md_eval
# The relative_md array now holds the relative Mahalanobis distances for test data.


min_dist, total_time, preds = mahalanobis_distance(
    train_features=train_embeddings,
    train_labels=df_train['label'].to_numpy(),
    eval_features=test_embeddings,
    return_full=False
)


%pip install seaborn
import seaborn as sns
sns.displot(min_dist)

# Calculate the full Mahalanobis distance matrix
full_dist, total_time = mahalanobis_distance(
    train_embeddings,
    df_train['label'].values,
    test_embeddings, 
    return_full=True
)

# 'full_dist' now contains the full distance matrix between each test sample and each class centroid
# 'total_time' records the time taken for the computation


full_dist.shape

preds = np.argmin(full_dist, axis=1)

preds

df_test['label'].shape

%pip install scikit-learn
from sklearn.metrics import roc_auc_score, f1_score
f1_score(df_test['label'].tolist(), preds, average='macro')

df = pd.DataFrame()
df['label'] = df_test['label'].tolist()
df['min_dist'] = min_dist
df['pred'] = preds
df['label0_dist'] = full_dist[:, 0].tolist()
df['label1_dist'] = full_dist[:, 1].tolist()
# df['label2_dist'] = full_dist[:, 2].tolist()
# df['label3_dist'] = full_dist[:, 3].tolist()
# df['label4_dist'] = full_dist[:, 4].tolist()
df

from sklearn.metrics import confusion_matrix
confusion_matrix(df['label'].tolist(), df['pred'].tolist())

sns.displot(data=df, x='min_dist', hue='label')

sns.displot(data=df, x='min_dist', hue='pred')

sns.displot(data=df[df['label']!= 0], x='min_dist', hue='label')

df['correct'] = df.apply(lambda x: x['label'] == x['pred'], axis=1)

df[df['correct'] == 0]

sns.displot(data=df, x='min_dist', hue='correct')

sns.displot(data=df[df['correct'] == 0], x='min_dist', hue='label')

df['min_dist'].describe()

df[df['min_dist'] > 1500]['correct'].value_counts()

df[df['min_dist'] > 2000]['correct'].value_counts()

df[df['min_dist'] > 2500]['correct'].value_counts()

incorrect_ix_ = df[df['correct'] == 0].index.tolist()

df_test.iloc[incorrect_ix_]

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

all_embeddings = np.vstack((train_embeddings, test_embeddings))
all_labels = np.hstack((df_train['label'].tolist(), df_test['label'].tolist()))


pca_result_label = PCA(n_components=2).fit_transform(all_embeddings)
tsne_result_label = TSNE(n_components=2, perplexity=30, n_iter=300).fit_transform(all_embeddings)
plt.scatter(pca_result_label[:, 0], pca_result_label[:, 1], c=all_labels, cmap='coolwarm', alpha=0.5)

incorrect_ix = [i + train_embeddings.shape[0] for i in incorrect_ix_]

plt.scatter(pca_result_label[:, 0], pca_result_label[:, 1], c=all_labels,  alpha=0.1)
plt.scatter(pca_result_label[incorrect_ix, 0], pca_result_label[incorrect_ix, 1], c=[10]*len(incorrect_ix), alpha=0.8, marker='x', s=60)

plt.scatter(tsne_result_label[:, 0], tsne_result_label[:, 1], c=all_labels, cmap='coolwarm', alpha=0.1)
plt.scatter(tsne_result_label[incorrect_ix, 0], tsne_result_label[incorrect_ix, 1], c=[10]*len(incorrect_ix), alpha=0.8, marker='x', s=60)

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC



# Label the embeddings: 0 for training, 1 for testing
train_labels = np.zeros((train_embeddings.shape[0],))
test_labels = np.ones((test_embeddings.shape[0],))

# Concatenate the embeddings and labels
all_embeddings = np.vstack((train_embeddings, test_embeddings))
all_labels = np.hstack((train_labels, test_labels))


pca_result = PCA(n_components=2).fit_transform(all_embeddings)
tsne_result = TSNE(n_components=2, perplexity=30, n_iter=300).fit_transform(all_embeddings)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=all_labels, cmap='coolwarm', alpha=0.5)

plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=all_labels, cmap='coolwarm', alpha=0.5)



# # It's a good practice to scale your data
# scaler = StandardScaler()
# all_embeddings_scaled = scaler.fit_transform(all_embeddings)

# Split data for training and validation of the logistic regression model
X_train, X_val, y_train, y_val = train_test_split(
    all_embeddings, all_labels, test_size=0.2, random_state=42)

# Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Support Vector Machine': SVC(probability=True)  # Enable probability estimates
}
# Train and evaluate each model
for model_name, simple_model in models.items():
    simple_model.fit(X_train, y_train)
    # Predict on the validation set
    y_pred_prob = simple_model.predict_proba(X_val)[:, 1]
    y_pred = simple_model.predict(X_val)
    # Evaluate
    auc_roc = roc_auc_score(y_val, y_pred_prob)
    f1 = f1_score(y_val, y_pred)
    print(f'{model_name} AUC-ROC score: {auc_roc:.2f} F1 score: {f1:.2f}')




