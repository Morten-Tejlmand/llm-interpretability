import json
import numpy as np
import pandas as pd
import torch
import pickle
import lzma
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ============================================================
# CONFIG
# ============================================================
SEED = 42
np.random.seed(SEED)

FEATURE_CLUSTERS_PATH = "outputs/feature_clusters.pkl.xz"
TRAINING_DATA_PATH = "outputs/training_dataset.csv"
TOKENS_PATH = "outputs/tokens.pt"
TOP_POSITIONS_PATH = "outputs/top_positions.npy"

TRAIN_WITH_CLUSTERS_OUT = "outputs/training_dataset_with_clusters_regression.csv"
CLUSTER_RISK_OUT = "outputs/cluster_risk.json"
TOKEN_TO_CLUSTERS_OUT = "outputs/token_to_clusters.json"

# ============================================================
# LOAD CLUSTERING ARTIFACTS
# ============================================================
print("Loading clustering artifacts...")

# Load pickle or pickle.xz automatically
with lzma.open(FEATURE_CLUSTERS_PATH, "rb") as f:
    cluster_results_raw = pickle.load(f)

# Convert keys to ints
cand_features = sorted(int(k) for k in cluster_results_raw.keys())

df_training = pd.read_csv(TRAINING_DATA_PATH)
per_prompt_ppl_base = df_training["ppl_base"].values
per_prompt_run_collision = df_training["is_runtime_collision"].values.astype(np.bool_)
tokens = torch.load(TOKENS_PATH).cpu()
n_prompts, seq_len = tokens.shape

top_positions = np.load(TOP_POSITIONS_PATH)

print(f"Loaded: {n_prompts} prompts, sequence length {seq_len}")

# ========================================================
# CLUSTER RISK COMPUTATION
# ========================================================

cluster_risk = {}
for f in cand_features:
    res = cluster_results_raw[f]
    labels = np.array(res["cluster_labels"], dtype=np.int32)
    pos_idx = top_positions[:, f]
    prompt_idx = (pos_idx // seq_len).astype(np.int32)

    for cl in np.unique(labels):
        mask = labels == cl
        prompts_for_cluster = prompt_idx[mask]
        unique_prompts = np.unique(prompts_for_cluster)
        if len(unique_prompts) == 0:
            continue

        risk = float(per_prompt_ppl_base[unique_prompts].mean())

        cluster_risk[(f, int(cl))] = {
            "risk": risk,
            "n_prompts": int(len(unique_prompts)),
            "n_positions": int(mask.sum()),
        }

print(f"Risks computed for {len(cluster_risk)} clusters.")

# Save risk table
export = {}
for (f, cl), v in cluster_risk.items():
    export.setdefault(str(f), {})[str(cl)] = v

with open(CLUSTER_RISK_OUT, "w") as f:
    json.dump(export, f, indent=2)

# ========================================================
# PROMPT-LEVEL CLUSTER FEATURES
# ========================================================
print("Building prompt-level cluster features...")

prompt_risk_values = [[] for _ in range(n_prompts)]
prompt_total_hits = np.zeros(n_prompts, dtype=np.int32)

for f in cand_features:
    res = cluster_results_raw[f]
    labels = np.array(res["cluster_labels"], dtype=np.int32)
    pos_idx = top_positions[:, f]
    prompt_idx = (pos_idx // seq_len).astype(np.int32)

    for j, cl in enumerate(labels):
        key = (f, int(cl))
        if key not in cluster_risk:
            continue
        p_idx = prompt_idx[j]
        prompt_risk_values[p_idx].append(cluster_risk[key]["risk"])
        prompt_total_hits[p_idx] += 1

mean_cluster_risk = np.zeros(n_prompts, dtype=np.float32)
max_cluster_risk = np.zeros(n_prompts, dtype=np.float32)
n_clusters_in_prompt = prompt_total_hits.copy()

for i in range(n_prompts):
    values = prompt_risk_values[i]
    if not values:
        mean_cluster_risk[i] = 0
        max_cluster_risk[i] = 0
    else:
        arr = np.array(values)
        mean_cluster_risk[i] = arr.mean()
        max_cluster_risk[i] = arr.max()

# ============================================================
# EXTRA FEATURES (requested)
# ============================================================

# ------------- 1.a Standard deviation of cluster risk -------------
std_cluster_risk = np.zeros(n_prompts, dtype=np.float32)

# ------------- 1.b Median cluster risk -------------
median_cluster_risk = np.zeros(n_prompts, dtype=np.float32)

# For cluster uniqueness
clusters_seen = [set() for _ in range(n_prompts)]   # 2.a
features_seen = [set() for _ in range(n_prompts)]   # 2.b

# We also need prompt lengths (token-based) 3.a
prompt_length_tokens = (tokens != 0).sum(dim=1).cpu().numpy()

# For polysemantic ratio 4.c:
# We approximate polysemantic activation ratio as:
#    risky_hits / total_top_positions_hits
total_positions_per_prompt = np.zeros(n_prompts, dtype=np.int32)
risky_positions_per_prompt = np.zeros(n_prompts, dtype=np.int32)

for f in cand_features:
    res = cluster_results_raw[f]
    labels = np.array(res["cluster_labels"], dtype=np.int32)
    pos_idx = top_positions[:, f]
    prompt_idx = (pos_idx // seq_len).astype(np.int32)

    for j, cl in enumerate(labels):
        p = prompt_idx[j]
        total_positions_per_prompt[p] += 1

        key = (f, int(cl))
        if key in cluster_risk:
            risky_positions_per_prompt[p] += 1
            clusters_seen[p].add(key)   # unique cluster (2.a)
            features_seen[p].add(f)     # unique feature  (2.b)

# compute std + median cluster risks
for i in range(n_prompts):
    vals = prompt_risk_values[i]
    if vals:
        arr = np.array(vals, dtype=np.float32)
        std_cluster_risk[i] = arr.std()
        median_cluster_risk[i] = np.median(arr)
    else:
        std_cluster_risk[i] = 0
        median_cluster_risk[i] = 0

# 2.a number of unique clusters
n_unique_clusters = np.array([len(s) for s in clusters_seen], dtype=np.int32)

# 2.b number of unique features
n_unique_features = np.array([len(s) for s in features_seen], dtype=np.int32)

# 2.c ratio normalisation features
hits_per_token = n_clusters_in_prompt / (prompt_length_tokens + 1e-9)
unique_clusters_per_token = n_unique_clusters / (prompt_length_tokens + 1e-9)

# 4.c polysemantic ratio
polysemantic_ratio = risky_positions_per_prompt / (total_positions_per_prompt + 1e-9)


df_training["mean_cluster_risk"] = mean_cluster_risk
df_training["max_cluster_risk"] = max_cluster_risk
df_training["n_clusters_in_prompt"] = n_clusters_in_prompt
df_training["std_cluster_risk"] = std_cluster_risk
df_training["median_cluster_risk"] = median_cluster_risk
df_training["n_unique_clusters"] = n_unique_clusters
df_training["n_unique_features"] = n_unique_features
df_training["hits_per_token"] = hits_per_token
df_training["unique_clusters_per_token"] = unique_clusters_per_token
df_training["prompt_length_tokens"] = prompt_length_tokens
df_training["polysemantic_ratio"] = polysemantic_ratio

df_training.to_csv(TRAIN_WITH_CLUSTERS_OUT, index=False)
print(f"Saved with-cluster features to {TRAIN_WITH_CLUSTERS_OUT}")

# ========================================================
# TRAIN REGRESSION MODEL
# ========================================================
print("Training regression model to predict perplexity...")

X = np.vstack([
    mean_cluster_risk,
    max_cluster_risk,
    n_clusters_in_prompt,
    std_cluster_risk,
    median_cluster_risk,
    n_unique_clusters,
    n_unique_features,
    hits_per_token,
    unique_clusters_per_token,
    prompt_length_tokens,
    polysemantic_ratio,
]).T
y = per_prompt_run_collision

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=SEED,
    max_depth=12,
    min_samples_leaf=3,
    class_weight="balanced"
)

model.fit(x_train, y_train)
probs = model.predict_proba(x_test)[:, 1]
preds = (probs > 0.5).astype(int)

# ========================================================
# CLASSIFICATION METRICS
# ========================================================
acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds, zero_division=0)
rec = recall_score(y_test, preds, zero_division=0)
f1 = f1_score(y_test, preds, zero_division=0)
auc = roc_auc_score(y_test, probs)

print("\n=== CLASSIFICATION METRICS ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")

# Save model
import joblib
joblib.dump(model, "outputs/runtime_collision_classifier.joblib")
print("Saved model to outputs/runtime_collision_classifier.joblib")

# y_test and preds are in test-split order,
# so we must recover the original prompt indices used in the split.

test_indices = y_test.index if isinstance(y_test, pd.Series) else None
# If y_test is a numpy array, generate the mapping manually:
if test_indices is None:
    # Re-run split but return indices
    _, X_test_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=SEED
    )
    test_indices = X_test_idx

# Build DataFrame
df_pred = pd.DataFrame({
    "prompt_index": test_indices,
    "true_target": y_test,
    "predicted_label": preds,
    "predicted_prob": probs,
})

# Add global metrics duplicated across rows for easy logging/analysis
df_pred["accuracy"] = acc
df_pred["precision"] = prec
df_pred["recall"] = rec
df_pred["f1"] = f1
df_pred["auc"] = auc

# Save
df_pred.to_csv("outputs/runtime_collision_predictions.csv", index=False)
print("Saved detailed prediction table to outputs/runtime_collision_predictions.csv")
