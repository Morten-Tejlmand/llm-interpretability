print("running optimized version...")

import os
import random
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
import pyarrow.parquet as pq
import pyarrow as pa
import json

# =========================
# CONFIG
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if device == "cuda" else torch.float32

MODEL_NAME = "meta-llama/Llama-3.2-1B"
SAE_RELEASE = "seonglae/Llama-3.2-1B-sae"
SAE_ID = "Llama-3.2-1B_blocks.12.hook_resid_pre_14336_topk_48_0.0002_49_fineweb_512"
TARGET_LAYER = 12

# runtime tuning
N_SAMPLES = 100
TOPK_PER_FEATURE = 250
CANDIDATE_FEATURES_FOR_ENTROPY = 2000
MAX_POSITIONS_FOR_COS_SIM = 8000

os.makedirs("outputs", exist_ok=True)

# =========================
# LOAD DATA SET OF TEXTS
# =========================
df_wiki = pl.read_csv("wikipedia_data.csv")
df_wiki = df_wiki.filter(pl.col("text").str.len_chars() > 10)

pool = [row["text"] for row in df_wiki.to_dicts()]
if len(pool) == 0:
    raise ValueError("wikipedia_data.csv contains no valid text rows.")

samples = random.sample(pool, min(N_SAMPLES, len(pool)))
prompts = [
    f"Context:\n{s}\n\nQuestion: Summarize the above text.\nAnswer:"
    for s in samples
]

# =========================
# LOAD MODEL + SAE
# =========================
login(token=os.getenv("HUGGINGFACE_TOKEN"))

model = HookedTransformer.from_pretrained_no_processing(
    model_name=MODEL_NAME, device=device, dtype=DTYPE
)
model.eval()

sae, cfg_dict, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
    SAE_RELEASE,
    SAE_ID,
    device=device,
)
sae = sae.to(dtype=DTYPE)

# =========================
# TOKENIZE
# =========================
tokens = model.to_tokens(prompts).to(device)
labels = tokens.clone().to(device)
pad_id = model.tokenizer.pad_token_id or 0

# =========================
# BASELINE FORWARD (hook only target layer)
# =========================
with torch.no_grad():
    logits, cache = model.run_with_cache(
        tokens,
        return_type="logits",
        names_filter=[f"blocks.{TARGET_LAYER}.hook_resid_pre"]
    )

# Move logits + labels to CPU
logits = logits.to("cpu")
labels = labels.to("cpu")

shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
mask = shift_labels != pad_id

def xent_masked(shift_logits, shift_labels, mask):
    logp = F.log_softmax(shift_logits, dim=-1)
    nll = -logp.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    nll = nll * mask.float()
    return nll.sum() / mask.sum().clamp_min(1)

base_loss = xent_masked(shift_logits, shift_labels, mask)
ppl_base = torch.exp(base_loss).item()
print(f"Baseline Perplexity: {ppl_base:.4f}")

# =========================
# EXTRACT HOOKED ACTIVATION
# =========================
acts = cache[f"blocks.{TARGET_LAYER}.hook_resid_pre"].detach().cpu()
del cache
torch.cuda.empty_cache()

with torch.no_grad():
    sae_cpu = sae.to("cpu")
    feature_acts = sae_cpu.encode(acts)
    sae = sae.to(device=device, dtype=DTYPE)
del sae_cpu

feature_acts_flat = feature_acts.reshape(-1, feature_acts.size(-1))

# =========================
# POLY SCORE FROM DECODER SPARSITY
# =========================
W_dec = sae.W_dec.T.detach().cpu()
decoder_l2 = torch.norm(W_dec, p=2, dim=0)
decoder_l1 = torch.norm(W_dec, p=1, dim=0)
poly_score_decoder = decoder_l1 / (decoder_l2 + 1e-12)

nF = feature_acts.size(-1)
cand_idx = torch.topk(poly_score_decoder, k=min(CANDIDATE_FEATURES_FOR_ENTROPY, nF)).indices

# =========================
# CLUSTERING ON TOP-K POSITIONS (AND SAVE CLUSTERS)
# =========================
tokens_flat = tokens.reshape(-1).detach().cpu().numpy().astype(np.int32)
token_embeddings = model.W_E.detach().to(torch.float32).cpu().numpy()

cluster_results = {}

def compute_clusters_for_feature(feature_id, top_positions):
    pos_idx = top_positions[:, feature_id]
    ids = tokens_flat[pos_idx]
    X = token_embeddings[ids]

    clustering = AgglomerativeClustering(
        n_clusters=None,
        linkage="average",
        distance_threshold=0.25
    )
    clustering.fit(X)

    # Decode token strings
    tokens_text = [model.tokenizer.decode([tid]) for tid in ids]

    # Build cluster->tokens mapping
    cluster_dict = {}
    for label in set(clustering.labels_):
        idxs = np.where(clustering.labels_ == label)[0]
        cluster_dict[int(label)] = [tokens_text[j] for j in idxs]

    # Compute centroids
    centroids = []
    for label in set(clustering.labels_):
        idxs = np.where(clustering.labels_ == label)[0]
        centroids.append(X[idxs].mean(axis=0).tolist())

    return {
        "feature_id": int(feature_id),
        "token_ids": ids.tolist(),
        "token_strings": tokens_text,
        "cluster_labels": clustering.labels_.tolist(),
        "n_clusters": int(clustering.n_clusters_),
        "clusters_to_tokens": cluster_dict,
        "cluster_centroids": centroids,
    }

# Precompute top positions for each feature
top_positions = torch.topk(
    feature_acts_flat,
    k=min(TOPK_PER_FEATURE, feature_acts_flat.size(0)),
    dim=0
).indices.cpu().numpy().astype(np.int32)

print("Running clustering for all candidate features...")

for f in cand_idx.tolist():
    cluster_results[f] = compute_clusters_for_feature(f, top_positions)

# Save clusters to JSON
with open("outputs/feature_clusters.json", "w") as f:
    json.dump(cluster_results, f)

print(f"Saved clusters for {len(cluster_results)} features to outputs/feature_clusters.json")

# =========================
# CLUSTER-BASED POLY SCORE
# =========================
poly_score_cluster = torch.zeros(nF, dtype=torch.float32)
for f in cand_idx.tolist():
    poly_score_cluster[f] = cluster_results[f]["n_clusters"]

# =========================
# SELECT FEATURES BY THRESHOLD
# =========================
scores = poly_score_cluster.float()
valid_scores = scores[scores > 0]

threshold_high = torch.quantile(valid_scores, 0.90)
threshold_low = torch.quantile(valid_scores, 0.10)

top_poly = torch.nonzero(scores >= threshold_high).squeeze(-1)
low_poly = torch.nonzero(scores <= threshold_low).squeeze(-1)

print(f"Selected {len(top_poly)} high-poly features")
print(f"Selected {len(low_poly)} low-poly features")

# =========================
# HOOK ABLATION
# =========================
@torch.no_grad()
def ablate_features(value, indices):
    z = sae.encode(value)
    z[..., indices] = 0
    recon = sae.decode(z)
    return value - sae.decode(z) + recon

def make_hook(indices):
    def _hook(value, hook):
        return ablate_features(value, indices)
    return _hook

hooks_poly = [(f"blocks.{TARGET_LAYER}.hook_resid_pre", make_hook(top_poly.to(device)))]
hooks_mono = [(f"blocks.{TARGET_LAYER}.hook_resid_pre", make_hook(low_poly.to(device)))]

# =========================
# RUN INTERVENTIONS
# =========================
with torch.no_grad():
    logits_poly = model.run_with_hooks(tokens, fwd_hooks=hooks_poly, return_type="logits")
logits_poly = logits_poly.to("cpu")

with torch.no_grad():
    logits_mono = model.run_with_hooks(tokens, fwd_hooks=hooks_mono, return_type="logits")
logits_mono = logits_mono.to("cpu")

# =========================
# PER-PROMPT METRICS
# =========================
def per_prompt_loss_fn(logits_alt):
    logits_alt = logits_alt.to(shift_labels.device)
    shift_alt = logits_alt[:, :-1, :]
    logp = F.log_softmax(shift_alt, dim=-1)
    nll = -logp.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    nll = nll * mask.float()
    return nll.sum(dim=1) / mask.sum(dim=1).clamp_min(1)

per_prompt_loss_base = per_prompt_loss_fn(logits)
per_prompt_loss_poly = per_prompt_loss_fn(logits_poly)
per_prompt_loss_mono = per_prompt_loss_fn(logits_mono)

per_prompt_ppl_base = torch.exp(per_prompt_loss_base).cpu().numpy()
per_prompt_ppl_poly = torch.exp(per_prompt_loss_poly).cpu().numpy()
per_prompt_ppl_mono = torch.exp(per_prompt_loss_mono).cpu().numpy()

delta_ppl_poly_per_prompt = per_prompt_ppl_poly - per_prompt_ppl_base
delta_ppl_mono_per_prompt = per_prompt_ppl_mono - per_prompt_ppl_base

# =========================
# FEATURE STRENGTH
# =========================
with torch.no_grad():
    poly_feature_strength = (
        feature_acts[..., top_poly]
        .abs()
        .sum(dim=-1)
        .to(torch.float32)
        .cpu()
        .numpy()
    )
    mono_feature_strength = (
        feature_acts[..., low_poly]
        .abs()
        .sum(dim=-1)
        .to(torch.float32)
        .cpu()
        .numpy()
    )

poly_strength_per_prompt = poly_feature_strength.mean(axis=1)
mono_strength_per_prompt = mono_feature_strength.mean(axis=1)

# =========================
# SAVE LOGITS
# =========================
torch.save(logits, "outputs/logits_baseline.pt")
torch.save(logits_poly, "outputs/logits_poly.pt")
torch.save(logits_mono, "outputs/logits_mono.pt")

# =========================
# SAVE TRAINING DATASET
# =========================
df_training = pd.DataFrame({
    "prompt": prompts,
    "poly_strength": poly_strength_per_prompt,
    "mono_strength": mono_strength_per_prompt,
    "ppl_base": per_prompt_ppl_base,
    "ppl_poly": per_prompt_ppl_poly,
    "ppl_mono": per_prompt_ppl_mono,
    "delta_ppl_poly": delta_ppl_poly_per_prompt,
    "delta_ppl_mono": delta_ppl_mono_per_prompt,
})

df_training.to_csv("outputs/training_dataset.csv", index=False)
print("Saved per-prompt training data to outputs/training_dataset.csv")

# =========================
# GLOBAL SUMMARY
# =========================
def mean_kl(p_ref, p_alt):
    kl = torch.sum(p_ref * (torch.log(p_ref + 1e-9) - torch.log(p_alt + 1e-9)), dim=-1)
    return kl.mean().item()

def cos_sim(ref_logits, alt_logits, max_positions=MAX_POSITIONS_FOR_COS_SIM):
    min_len = min(ref_logits.size(1), alt_logits.size(1))
    ref = ref_logits[:, :min_len, :].reshape(-1, ref_logits.size(-1))
    alt = alt_logits[:, :min_len, :].reshape(-1, alt_logits.size(-1))
    idx = torch.randperm(ref.size(0), device=ref.device)[:max_positions]
    return F.cosine_similarity(ref[idx], alt[idx], dim=-1).mean().item()

p_base = softmax(shift_logits, dim=-1)
p_poly = softmax(logits_poly[:, :-1, :], dim=-1)
p_mono = softmax(logits_mono[:, :-1, :], dim=-1)

df_summary = pd.DataFrame({
    "Condition": ["Baseline", "Remove Polysemantic", "Remove Monosemantic"],
    "Next-token Loss (NLL)": [
        float(base_loss),
        float(per_prompt_loss_poly.mean()),
        float(per_prompt_loss_mono.mean()),
    ],
    "Perplexity": [
        ppl_base,
        float(per_prompt_ppl_poly.mean()),
        float(per_prompt_ppl_mono.mean()),
    ],
    "Δ Perplexity": [
        0.0,
        float(per_prompt_ppl_poly.mean() - ppl_base),
        float(per_prompt_ppl_mono.mean() - ppl_base),
    ],
    "Mean KL (vs Base)": [
        0.0,
        mean_kl(p_base, p_poly),
        mean_kl(p_base, p_mono),
    ],
    "Cosine Sim (vs Base)": [
        1.0,
        cos_sim(shift_logits, logits_poly),
        cos_sim(shift_logits, logits_mono),
    ],
})

print("\n=== Global Summary ===")
print(df_summary.to_string(index=False))
df_summary.to_csv("outputs/summary_metrics.csv", index=False)
print("Saved summary table to outputs/summary_metrics.csv")


# =========================
# PER-PROMPT POLYSEMANTIC IMPACT
# =========================

df_pp = pd.DataFrame({
    "prompt": prompts,
    "ppl_base": per_prompt_ppl_base,
    "ppl_poly": per_prompt_ppl_poly,
    "delta_ppl_poly": delta_ppl_poly_per_prompt,
})

# Sort highest degradation first
df_pp_sorted = df_pp.sort_values("delta_ppl_poly", ascending=False)

print("\n=== Per-Prompt Impact of Removing Polysemantic Features (Top 15 worst) ===")
print(df_pp_sorted[["delta_ppl_poly", "ppl_base", "ppl_poly"]].head(15).to_string())

# Save full table
df_pp_sorted.to_csv("outputs/per_prompt_poly_impact.csv", index=False)
print("Saved detailed per-prompt impact table to outputs/per_prompt_poly_impact.csv")

print("\n=== Worst prompt text ===")
print(df_pp_sorted.iloc[0]["prompt"])
