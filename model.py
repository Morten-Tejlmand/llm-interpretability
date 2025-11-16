import os
import math
import random
import pickle
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn.functional as F
from torch.nn.functional import softmax
from dotenv import load_dotenv
from huggingface_hub import login
from transformer_lens import HookedTransformer
from sae_lens import SAE
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import pyarrow.parquet as pq
import pyarrow as pa

# classification thresholding for "high-poly" label
HIGH_POLY_METHOD = "percentile"  # "percentile" or "absolute"
HIGH_POLY_PERCENTILE = 75        # if percentile method
HIGH_POLY_ABS_THRESHOLD = None   # if absolute method, e.g., 50.0

# TF-IDF configuration
TFIDF_MAX_FEATURES = 8000
TEST_SIZE = 0.2
RANDOM_STATE = 1337
# =========================
# TRAIN BASELINE ML MODELS
#   1) Regression: predict poly_strength from TF-IDF(prompt)
#   2) Classification: predict high_poly (binary) from TF-IDF(prompt)
# =========================

# Vectorize prompts
vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
X = vectorizer.fit_transform(df_training["prompt"])

# -------- Regression target --------
y_reg = df_training["poly_strength"].values.astype(float)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
reg = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
reg.fit(X_train, y_train)
reg_r2 = reg.score(X_test, y_test)
print(f"[Regression] R^2 on holdout: {reg_r2:.4f}")

# -------- Classification label --------
if HIGH_POLY_METHOD == "percentile":
    thresh = np.percentile(df_training["poly_strength"].values, HIGH_POLY_PERCENTILE)
elif HIGH_POLY_METHOD == "absolute":
    if HIGH_POLY_ABS_THRESHOLD is None:
        raise ValueError("Set HIGH_POLY_ABS_THRESHOLD for absolute method.")
    thresh = HIGH_POLY_ABS_THRESHOLD
else:
    raise ValueError("HIGH_POLY_METHOD must be 'percentile' or 'absolute'.")

y_cls = (df_training["poly_strength"].values >= thresh).astype(int)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X, y_cls, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_cls
)
# simple, fast baseline
clf = LogisticRegression(max_iter=2000, n_jobs=None)
clf.fit(Xc_train, yc_train)
cls_acc = clf.score(Xc_test, yc_test)
print(f"[Classification] Accuracy on holdout: {cls_acc:.4f} (threshold={thresh:.4f})")

# =========================
# SAVE ARTIFACTS
# =========================
with open("outputs/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("outputs/poly_strength_regressor.pkl", "wb") as f:
    pickle.dump(reg, f)
with open("outputs/high_poly_classifier.pkl", "wb") as f:
    pickle.dump(clf, f)

# Save metadata
meta = {
    "high_poly_method": HIGH_POLY_METHOD,
    "high_poly_percentile": HIGH_POLY_PERCENTILE,
    "high_poly_abs_threshold": HIGH_POLY_ABS_THRESHOLD,
    "derived_threshold": float(thresh),
    "tfidf_max_features": TFIDF_MAX_FEATURES,
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "top_n_features": TOP_N,
    "target_layer": TARGET_LAYER,
}
pd.Series(meta).to_json("outputs/model_meta.json", indent=2)

print("\nArtifacts saved in ./outputs/")
print(" - training_dataset.csv")
print(" - tfidf_vectorizer.pkl")
print(" - poly_strength_regressor.pkl")
print(" - high_poly_classifier.pkl")
print(" - model_meta.json")
print(" - summary_metrics.csv")
print(" - feature_scores.csv")
print(" - logits_baseline.pt / logits_poly.pt / logits_mono.pt")
print(" - top_poly.npy / low_poly.npy")
print(" - activations_top_tokens.parquet")
