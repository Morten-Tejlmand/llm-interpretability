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
import pickle
import lzma

# config
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if device == "cuda" else torch.float32

# MODEL_NAME = "meta-llama/Llama-3.2-1B"                                                      
# SAE_RELEASE = "seonglae/Llama-3.2-1B-sae"
#SAE_ID = "Llama-3.2-1B_blocks.12.hook_resid_pre_14336_topk_48_0.0002_49_fineweb_512"
MODEL_NAME = 'gpt2-small'
SAE_RELEASE = "gpt2-small-resid-post-v5-32k"
SAE_ID = 'blocks.0.hook_resid_post'

TARGET_LAYER = 0

N_SAMPLES = 100
TOPK_PER_FEATURE = 250
CANDIDATE_FEATURES_FOR_ENTROPY = 2000
MAX_POSITIONS_FOR_COS_SIM = 8000

os.makedirs("outputs", exist_ok=True)

def sample_with_ratio(df, ratio=0.7, target_col="is_runtime_collision", total=1000, seed=42):
    """
    Returns a sample with class ratio = (ratio : 1-ratio).
    ratio = proportion of class 1 (runtime collision).
    total = total number of samples in output.

    Automatically oversamples the minority class when needed.
    """
    np.random.seed(seed)

    target1 = int(total * ratio)
    target0 = total - target1

    df0 = df[df[target_col] == 0]
    df1 = df[df[target_col] == 1]

    s0 = df0.sample(n=target0, replace=(len(df0) < target0), random_state=seed)
    s1 = df1.sample(n=target1, replace=(len(df1) < target1), random_state=seed)

    return pd.concat([s0, s1], ignore_index=True).sample(frac=1, random_state=seed)


df_wiki = pl.read_csv("runtime_collision_results/gpt2-small/runtime_collision_prompts.csv")
df_wiki_pd = df_wiki.to_pandas()

sample_rows = sample_with_ratio(df_wiki_pd, ratio=0.65, total=N_SAMPLES, seed=SEED)

prompts = [
    f"Context:\n{s}\n\nQuestion: Summarize the above text.\nAnswer:"
    for s in sample_rows["text"]
]


# load env variables
#login(token="")

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

# tokenize
tokens = model.to_tokens(prompts).to(device)
labels = tokens.clone().to(device)
pad_id = model.tokenizer.pad_token_id or 0

# save tokens for later risk modeling
torch.save(tokens.cpu(), "outputs/tokens.pt")

# baseline logits
with torch.no_grad():
    logits, cache = model.run_with_cache(
        tokens,
        return_type="logits",
        names_filter=[f"blocks.{TARGET_LAYER}.hook_resid_pre"]
    )
# move logits + labels to cpu
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

# extract hooked activation
acts = cache[f"blocks.{TARGET_LAYER}.hook_resid_pre"].detach().cpu()
del cache
torch.cuda.empty_cache()

with torch.no_grad():
    sae_cpu = sae.to("cpu")
    feature_acts = sae_cpu.encode(acts)
    sae = sae.to(device=device, dtype=DTYPE)
del sae_cpu

feature_acts_flat = feature_acts.reshape(-1, feature_acts.size(-1))

# poly score from decoder sparsity
W_dec = sae.W_dec.T.detach().cpu()
decoder_l2 = torch.norm(W_dec, p=2, dim=0)
decoder_l1 = torch.norm(W_dec, p=1, dim=0)
poly_score_decoder = decoder_l1 / (decoder_l2 + 1e-12)

nF = feature_acts.size(-1)
cand_idx = torch.topk(poly_score_decoder, k=min(CANDIDATE_FEATURES_FOR_ENTROPY, nF)).indices

# clustering on top-k activations
tokens_flat = tokens.reshape(-1).detach().cpu().numpy().astype(np.int32)
token_embeddings = model.W_E.detach().to(torch.float32).cpu().numpy()

cluster_results = {}
print("Clustering top features...")
def compute_clusters_for_feature(feature_id, top_positions):
    pos_idx = top_positions[:, feature_id]
    ids = tokens_flat[pos_idx]
    X = token_embeddings[ids]

    clustering = AgglomerativeClustering(
        n_clusters=None,
        linkage="average",
        distance_threshold=0.18,
        metric="cosine",
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
        "n_clusters": int(clustering.n_clusters_),
        "cluster_centroids": centroids,
        "cluster_labels": clustering.labels_.tolist()

    }

# Precompute top positions for each feature
top_positions = torch.topk(
    feature_acts_flat,
    k=min(TOPK_PER_FEATURE, feature_acts_flat.size(0)),
    dim=0
).indices.cpu().numpy().astype(np.int32)

np.save("outputs/top_positions.npy", top_positions)


for f in cand_idx.tolist():
    cluster_results[f] = compute_clusters_for_feature(f, top_positions)

with lzma.open("outputs/feature_clusters.pkl.xz", "wb") as f:
    pickle.dump(cluster_results, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saved clusters for {len(cluster_results)} features to outputs/feature_clusters.json")

print("Computing per-prompt metrics...")
# PER-PROMPT METRICS
def per_prompt_loss_fn(logits_alt):
    logits_alt = logits_alt.to(shift_labels.device)
    shift_alt = logits_alt[:, :-1, :]
    logp = F.log_softmax(shift_alt, dim=-1)
    nll = -logp.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    nll = nll * mask.float()
    return nll.sum(dim=1) / mask.sum(dim=1).clamp_min(1)


per_prompt_loss_base = per_prompt_loss_fn(logits)


per_prompt_ppl_base = torch.exp(per_prompt_loss_base).cpu().numpy()

# SAVE LOGITS
torch.save(logits, "outputs/logits_baseline.pt")

# SAVE TRAINING DATASET
df_training = pd.DataFrame({
    "prompt": prompts,
    "ppl_base": per_prompt_ppl_base,
    "original_index": sample_rows.index.values,
    "is_runtime_collision": sample_rows["is_runtime_collision"].values,
})


df_training.to_csv("outputs/training_dataset.csv", index=False)
print("Saved training dataset to outputs/training_dataset.csv")
print("doing summary")
# GLOBAL SUMMARY
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

df_summary = pd.DataFrame({
    "Condition": ["Baseline"],
    "Next-token Loss (NLL)": [float(base_loss)],
    "Perplexity": [ppl_base],
})

print("\n=== Global Summary ===")
print(df_summary.to_string(index=False))
df_summary.to_csv("outputs/summary_metrics.csv", index=False)
print("Saved summary table to outputs/summary_metrics.csv")


# PER-PROMPT POLYSEMANTIC IMPACT
df_pp = pd.DataFrame({
    "prompt": prompts,
    "ppl_base": per_prompt_ppl_base,
    "original_index": sample_rows.index.values,
    "is_runtime_collision": sample_rows["is_runtime_collision"].values,
})


df_pp_sorted_base = df_pp.sort_values("ppl_base", ascending=False)

print("\n=== Prompts with Highest Base Perplexity (Top 15) ===")
print(df_pp_sorted_base[["ppl_base"]].head(15).to_string())

print("\n=== Worst base perplexity prompt text ===")
print(df_pp_sorted_base.iloc[0]["prompt"])



df_pp_sorted_base = df_pp.sort_values("ppl_base", ascending=True)

print("\n=== Prompts with Lowest Base Perplexity (Top 15) ===")
print(df_pp_sorted_base[["ppl_base"]].head(15).to_string())

print("\n=== Worst base perplexity prompt text ===")
print(df_pp_sorted_base.iloc[0]["prompt"])
