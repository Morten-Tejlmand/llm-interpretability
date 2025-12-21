print("Starting preds.py")
import json
import numpy as np
import pandas as pd
import torch
import pickle
import lzma
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


SEED = 42
np.random.seed(SEED)
TARGET_LAYER = 10
SAE_ID_SAVE = f"layer_{TARGET_LAYER}_16_k_gemma"

FEATURE_CLUSTERS_PATH = f"outputs/feature_clusters_{SAE_ID_SAVE}.pkl.xz"
TRAINING_DATA_PATH = f"outputs/training_dataset_{SAE_ID_SAVE}.csv"
TOKENS_PATH = f"outputs/tokens_{SAE_ID_SAVE}.pt"
TOP_POSITIONS_PATH = f"outputs/top_positions_{SAE_ID_SAVE}.npy"
PROMPT_LENGTHS_PATH = f"outputs/prompt_lengths_{SAE_ID_SAVE}.npy"
SUBSAMPLE_IDX_PATH = f"outputs/token_subsample_idx_{SAE_ID_SAVE}.npy"
CAND_IDX_PATH = f"outputs/cand_idx_{SAE_ID_SAVE}.npy"

FEATURE_ACTS_PATH = f"outputs/feature_acts_cand_{SAE_ID_SAVE}.pt"

TRAIN_WITH_CLUSTERS_OUT = f"outputs/training_dataset_with_clusters_regression_{SAE_ID_SAVE}.csv"
CLUSTER_RISK_OUT = f"outputs/cluster_risk_{SAE_ID_SAVE}.json"
MODEL_OUT = f"outputs/runtime_collision_xgboost_best_{SAE_ID_SAVE}.joblib"
PRED_OUT = f"results/runtime_collision_preds_with_proba_{SAE_ID_SAVE}.csv"

with lzma.open(FEATURE_CLUSTERS_PATH, "rb") as f:
    cluster_results_raw = pickle.load(f)

df_training = pd.read_csv("data/gpt_data/gpt-2-layer10.csv")
df_filtering = pd.read_csv("data/gpt_data/runtime_interference_units.csv")
n_prompts = len(df_training)

per_prompt_ppl_base = df_training["ppl_base"].values
df_training["label"] = df_training["label"].apply(lambda x: 1 if x == 'single' else 0)
units_for_filter = set(df_filtering["unit"].values)

df_training["label"] = (
    (df_training["label"] == 1) &
    (df_training["unit"].isin(units_for_filter))
).astype(int)

y = df_training["label"].values


prompt_lengths = np.load(PROMPT_LENGTHS_PATH)
sub_idx = np.load(SUBSAMPLE_IDX_PATH)

cand_idx = np.load(CAND_IDX_PATH).astype(np.int32)
feat_to_col = {int(fid): j for j, fid in enumerate(cand_idx.tolist())}

tokens = torch.load(TOKENS_PATH, weights_only=False)
if isinstance(tokens, np.ndarray):
    tokens = torch.from_numpy(tokens)
tokens = tokens.cpu()

top_positions = np.load(TOP_POSITIONS_PATH)

feature_acts_flat = (
    torch.load(FEATURE_ACTS_PATH, map_location="cpu")
    .float()
    .numpy()
)

token_to_prompt_full = np.repeat(
    np.arange(n_prompts),
    prompt_lengths
)
token_to_prompt = token_to_prompt_full[sub_idx]

cand_features = sorted(
    f for f in cluster_results_raw.keys()
    if int(f) in feat_to_col
)

n_features = len(cand_features)
n_tokens = tokens.numel()

units = df_training["unit"].values

rng = np.random.default_rng(SEED)

# One label per unit (majority vote)
unit_df = (
    df_training
    .groupby("unit")["label"]
    .mean()
    .reset_index()
)

# Binary unit label
unit_df["unit_label"] = (unit_df["label"] >= 0.5).astype(int)

train_units = set()
test_units = set()

for unit_label in [0, 1]:
    units_this_label = unit_df.loc[
        unit_df["unit_label"] == unit_label, "unit"
    ].values

    rng.shuffle(units_this_label)

    n_train = int(0.75 * len(units_this_label))

    train_units.update(units_this_label[:n_train])
    test_units.update(units_this_label[n_train:])

# Map back to row indices
units = df_training["unit"].values
idx_train = np.where(np.isin(units, list(train_units)))[0]
idx_test = np.where(np.isin(units, list(test_units)))[0]

assert set(units[idx_train]).isdisjoint(set(units[idx_test]))


is_train_prompt = np.zeros(n_prompts, dtype=bool)
is_train_prompt[idx_train] = True

is_train_token = is_train_prompt[token_to_prompt]
train_token_idx = np.flatnonzero(is_train_token)

is_train = np.zeros(n_prompts, dtype=bool)
is_train[idx_train] = True

cluster_risk = {}

for col, f in enumerate(cand_idx):
    f = int(f)

    if f not in cand_features:
        continue

    res = cluster_results_raw[f]
    labels = np.asarray(res["cluster_labels"], dtype=np.int32)

    pos_idx = top_positions[:, col]
    prompt_idx = token_to_prompt[pos_idx]

    train_mask = is_train[prompt_idx]
    labels_tr = labels[train_mask]
    prompt_idx_tr = prompt_idx[train_mask]

    for cl in np.unique(labels_tr):
        mask = labels_tr == cl
        prompts = np.unique(prompt_idx_tr[mask])

        if len(prompts) == 0:
            continue

        cluster_risk[(f, int(cl))] = {
            "risk": float(per_prompt_ppl_base[prompts].mean()),
            "n_prompts": int(len(prompts)),
            "n_positions": int(mask.sum()),
        }


# Save cluster risks
export = {}
for (f, cl), v in cluster_risk.items():
    export.setdefault(str(f), {})[str(cl)] = v

with open(CLUSTER_RISK_OUT, "w") as f:
    json.dump(export, f, indent=2)

print("Building prompt-level features...")

prompt_risk_values = [[] for _ in range(n_prompts)]
prompt_total_hits = np.zeros(n_prompts, dtype=np.int32)

cluster_keys_per_prompt = [[] for _ in range(n_prompts)]
features_seen = [set() for _ in range(n_prompts)]

total_positions_per_prompt = np.zeros(n_prompts, dtype=np.int32)
risky_positions_per_prompt = np.zeros(n_prompts, dtype=np.int32)

for f in cand_features:
    col = feat_to_col[f]
    res = cluster_results_raw[f]
    labels = np.asarray(res["cluster_labels"], dtype=np.int32)

    pos_idx = top_positions[:, col]
    prompt_idx = token_to_prompt[pos_idx]

    if len(labels) != len(pos_idx):
        raise ValueError(f"Length mismatch for feature {f}")

    for j, cl in enumerate(labels):
        p = int(prompt_idx[j])
        total_positions_per_prompt[p] += 1

        key = (f, int(cl))
        if key not in cluster_risk:
            continue

        r = cluster_risk[key]["risk"]
        activation = float(feature_acts_flat[pos_idx[j], col])

        prompt_risk_values[p].append(r * activation)
        prompt_total_hits[p] += 1
        risky_positions_per_prompt[p] += 1

        cluster_keys_per_prompt[p].append(key)
        features_seen[p].add(f)

cluster_entropy = np.zeros(n_prompts, dtype=np.float32)

for i in range(n_prompts):
    keys = cluster_keys_per_prompt[i]
    if not keys:
        continue

    # keys are (feature_id, cluster_id) tuples; entropy over these pairs
    arr = np.asarray(keys, dtype=np.int64)  # shape (N, 2)
    _, counts = np.unique(arr, axis=0, return_counts=True)

    probs = counts.astype(np.float32) / counts.sum()
    cluster_entropy[i] = -(probs * np.log(probs + 1e-9)).sum()
weighted_entropy = np.zeros(n_prompts, dtype=np.float32)
for i in range(n_prompts):
    vals = prompt_risk_values[i]
    if not vals:
        continue
    w = np.asarray(vals, dtype=np.float32)
    p = w / (w.sum() + 1e-9)
    weighted_entropy[i] = -(p * np.log(p + 1e-9)).sum()

def agg_stat(fn):
    out = np.zeros(n_prompts, dtype=np.float32)
    for i, vals in enumerate(prompt_risk_values):
        if vals:
            out[i] = fn(np.asarray(vals, dtype=np.float32))
    return out

mean_cluster_risk = agg_stat(np.mean)
max_cluster_risk = agg_stat(np.max)
std_cluster_risk = agg_stat(np.std)
median_cluster_risk = agg_stat(np.median)

risk_p90 = agg_stat(lambda x: np.percentile(x, 90))
risk_p95 = agg_stat(lambda x: np.percentile(x, 95))
risk_p99 = agg_stat(lambda x: np.percentile(x, 99))

risk_sum = np.zeros(n_prompts, dtype=np.float32)
risk_sum_log = np.zeros(n_prompts, dtype=np.float32)

for i, vals in enumerate(prompt_risk_values):
    if vals:
        s = float(np.sum(np.asarray(vals, dtype=np.float32)))
        risk_sum[i] = s
        risk_sum_log[i] = np.log1p(s)

dominance_ratio = np.zeros(n_prompts, dtype=np.float32)
for i, vals in enumerate(prompt_risk_values):
    if vals:
        arr = np.asarray(vals)
        dominance_ratio[i] = arr.max() / (arr.mean() + 1e-9)

n_unique_clusters = np.array([len(set(keys)) for keys in cluster_keys_per_prompt], dtype=np.int32)
n_unique_features = np.array([len(s) for s in features_seen], dtype=np.int32)


polysemantic_ratio = risky_positions_per_prompt / (total_positions_per_prompt + 1e-9)


df_training = df_training.assign(
    mean_cluster_risk=mean_cluster_risk,
    max_cluster_risk=max_cluster_risk,
    std_cluster_risk=std_cluster_risk,
    median_cluster_risk=median_cluster_risk,
    n_clusters_in_prompt=prompt_total_hits,
    n_unique_clusters=n_unique_clusters,
    n_unique_features=n_unique_features,
    polysemantic_ratio=polysemantic_ratio,
    cluster_entropy=cluster_entropy,
    weighted_entropy=weighted_entropy,
    dominance_ratio=dominance_ratio,
    risk_p95=risk_p95,
    risk_p99=risk_p99,
    risk_sum=risk_sum,
    risk_sum_log=risk_sum_log,
)

df_training.to_csv(TRAIN_WITH_CLUSTERS_OUT, index=False)
print(f"Saved features to {TRAIN_WITH_CLUSTERS_OUT}")


X = np.column_stack([
    mean_cluster_risk,
    max_cluster_risk,
    std_cluster_risk,
    median_cluster_risk,
    prompt_total_hits,
    n_unique_clusters,
    n_unique_features,
    polysemantic_ratio,
    cluster_entropy,
    weighted_entropy,
    dominance_ratio,
    risk_p95,
    risk_p99,
    risk_sum,
    risk_sum_log,
])

x_train, x_test = X[idx_train], X[idx_test]
y_train, y_test = y[idx_train], y[idx_test]

param_grid = {
    "n_estimators": [300, 600, 1000],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
}

xgb_model = XGBClassifier(
    colsample_bytree=0.9,
    random_state=SEED,
)

grid = GridSearchCV(
    xgb_model,
    param_grid,
    cv=2,
    n_jobs=-1,
    verbose=1,
)

grid.fit(x_train, y_train)
best_model = grid.best_estimator_

# Probabilities for the positive class (label==1, i.e. 'single' by your mapping)
proba_test = best_model.predict_proba(x_test)[:, 1]
proba_train = best_model.predict_proba(x_train)[:, 1]

# Threshold-independent metrics
roc_auc = roc_auc_score(y_test, proba_test)
pr_auc = average_precision_score(y_test, proba_test)

print("\n=== PROBABILITY-BASED METRICS (TEST) ===")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"PR  AUC: {pr_auc:.4f}")

def metrics_over_thresholds(y_true, y_proba, thresholds):
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(np.int32)

        rows.append({
            "threshold": float(t),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "pred_pos_rate": float(y_pred.mean()),
        })
    return pd.DataFrame(rows)

thresholds = np.round(np.linspace(0.05, 0.95, 19), 2)

df_thr_test = metrics_over_thresholds(y_test, proba_test, thresholds)
df_thr_train = metrics_over_thresholds(y_train, proba_train, thresholds)

print("\n=== THRESHOLD SWEEP (TEST) ===")
print(df_thr_test.to_string(index=False))


best_row_train = df_thr_train.sort_values(
    ["f1", "recall", "precision", "accuracy"],
    ascending=False
).iloc[0]
best_t = float(best_row_train["threshold"])


y_pred_test_best = (proba_test >= best_t).astype(np.int32)
cm_test = confusion_matrix(y_test, y_pred_test_best)

print("\n=== CONFUSION MATRIX (TEST, threshold picked on TRAIN) ===")
print(cm_test)

print("\n=== CLASSIFICATION REPORT TEST, threshold picked on TRAIN ===")
print(classification_report(y_test, y_pred_test_best, digits=4, zero_division=0))

# Optional: also show train report at that same threshold
y_pred_train_best = (proba_train >= best_t).astype(np.int32)
print("\n=== CLASSIFICATION REPORT TRAIN, same threshold ===")
print(classification_report(y_train, y_pred_train_best, digits=4, zero_division=0))


# Confusion matrix + report at best threshold
y_pred_best = (proba_test >= best_t).astype(np.int32)
cm = confusion_matrix(y_test, y_pred_best)

print("\n=== CONFUSION MATRIX (TEST, best threshold) ===")
print(cm)

print("\n=== CLASSIFICATION REPORT (TEST, best threshold) ===")
print(classification_report(y_test, y_pred_best, digits=4, zero_division=0))

df_test_preds = df_training.loc[idx_test].copy()

df_test_preds["label_proba"] = proba_test
df_test_preds["label_pred"] = y_pred_best

df_test_preds.to_csv(PRED_OUT, index=False)

print(f"\nSaved TEST predictions only to {PRED_OUT}")