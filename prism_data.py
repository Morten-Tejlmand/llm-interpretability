# CELL 1: CONFIG & IMPORTS

import os
import random
import itertools
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import glob

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from sae_lens import SAE
import requests


# ----------------- PATHS: EDIT THESE -----------------

# PRISM descriptions for GPT-2 small SAE
API_URL = ("https://api.github.com/repos/lkopf/prism/contents/descriptions/gemini-1-5-pro/"
    "gpt2-small-sae")

response = requests.get(API_URL)
files = response.json()

csv_urls = []

for f in files:
    if f["name"].endswith(".csv"):
        csv_urls.append(f["download_url"])


# PRISM polysemanticity + COSY metrics for GPT-2 small SAE
METRICS_CSV = (
    "https://raw.githubusercontent.com/lkopf/prism/refs/heads/main/results/"
    "meta-evaluation_cosine-similarity_target-gpt2-small-sae_textgen-gemini-1-5-pro_mean_evalgen-gemini-1-5-pro_cosmopedia_1000.csv"
)




# ----------------- MODELS -----------------

EVAL_MODEL_NAME = "gpt2-small"

# Can be the same family or something stronger. Start simple.
GEN_MODEL_NAME = "gpt2"

# SAE release used by PRISM for GPT-2 small: v5, width 32k
SAE_RELEASE = "callummcdougall/sae-gpt2-small-32k-v5"


# ----------------- EXPERIMENT KNOBS -----------------
MODEL_TAG = EVAL_MODEL_NAME.replace("/", "_")

# Where to save experiment results
OUTPUT_DIR = os.path.join("runtime_collision_results", MODEL_TAG)
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# How many different SAE features to analyse in this run
N_FEATURES_TO_TEST = 15

# How many descriptions (concepts) per feature to use at most
MAX_DESCRIPTIONS_PER_FEATURE = 5  # use all if small; cap if large

# How many samples per single concept (A, B, C, ...)
N_SAMPLES_PER_CONCEPT = 4

# How many samples per pair of concepts (AB, AC, ...)
N_SAMPLES_PER_PAIR = 4

# Max generation length for synthetic prompts
MAX_NEW_TOKENS = 128


# CELL 2: LOAD PRISM DESCRIPTIONS + METRICS FOR GPT-2 SMALL SAE


# 1) Load polysemanticity + description quality metrics
metrics_df = pd.read_csv(METRICS_CSV)
print("Metrics columns:", metrics_df.columns.tolist())
print("Number of feature rows in metrics:", len(metrics_df))

# 2) Load all description CSVs for gpt2-small-sae

desc_dfs = []
for path in csv_urls:
    df = pd.read_csv(path)
    # Expect columns: layer,unit,description,mean_activation,highlights (your example)
    desc_dfs.append(df)

descriptions_df = pd.concat(desc_dfs, ignore_index=True)
print("Descriptions columns:", descriptions_df.columns.tolist())
print("Number of description rows:", len(descriptions_df))

# 3) Merge descriptions with metrics on (layer, unit)
merged_df = descriptions_df.merge(metrics_df, on=["layer", "unit"], how="left")

print("Merged columns:", merged_df.columns.tolist())
print("Example merged rows:")
print(merged_df.head(5))

# Utility: list all features with at least 2 descriptions
feature_counts = merged_df.groupby(["layer", "unit"]).size().sort_values(ascending=False)
print("\nTop features by number of descriptions:")
print(feature_counts.head(10))


# CELL 3: SELECT FEATURES TO TEST AND PREP CONCEPT LISTS

# Group by (layer, unit)
grouped = merged_df.groupby(["layer", "unit"])

# Get candidate features with at least 2 descriptions
candidate_features = []
for (layer, unit), g in grouped:
    if len(g) >= 2:
        candidate_features.append((layer, unit))

print(f"Found {len(candidate_features)} features with >= 2 descriptions.")
print("First few candidate features:", candidate_features[:10])

# Take the first N_FEATURES_TO_TEST for this run
features_to_test = candidate_features[:N_FEATURES_TO_TEST]
print(f"\nWill test these features (layer, unit): {features_to_test}")

# Build a dict: (layer, unit) -> description rows + metrics
feature_concepts = {}

for (layer, unit) in features_to_test:
    g = grouped.get_group((layer, unit)).reset_index(drop=True)
    # Optionally subsample descriptions if there are many
    if len(g) > MAX_DESCRIPTIONS_PER_FEATURE:
        g = g.sample(n=MAX_DESCRIPTIONS_PER_FEATURE, random_state=SEED).reset_index(drop=True)
    feature_concepts[(layer, unit)] = g

# Quick inspection
for (layer, unit), df_feat in feature_concepts.items():
    print(f"\nFeature (layer={layer}, unit={unit}) with {len(df_feat)} descriptions:")
    for i, row in df_feat.iterrows():
        print(f"  concept[{i}] description: {row['description']}")
    print("  cosine_similarity:", df_feat['cosine_similarity'].iloc[0])
    print("  max_auc:", df_feat['max_auc'].iloc[0], "max_mad:", df_feat['max_mad'].iloc[0])


# CELL 4 (UPDATED): LOAD GPT-2 SMALL + SAE PER LAYER, MATCHING PRISM

# Load eval model once
eval_model = HookedTransformer.from_pretrained(
    EVAL_MODEL_NAME,
    device=device,
    dtype=torch.float32,
)
eval_model.eval()
print("Loaded eval model:", EVAL_MODEL_NAME)

# ---- SAE RELEASE PRISM USES FOR GPT-2 SMALL SAE v5 (32k, resid_post) ----
SAE_RELEASE = "gpt2-small-resid-post-v5-32k"

# Pre-compute, from merged_df, the max unit index per layer that PRISM uses
max_unit_by_layer = merged_df.groupby("layer")["unit"].max().to_dict()
print("Max PRISM unit index per layer:", max_unit_by_layer)

sae_cache = {}

def get_sae_for_layer(layer: int) -> SAE:
    """
    Load (and cache) the SAE for a given layer.

    Assumes the SAE in SAE_RELEASE is stored under
        sae_id = f"blocks.{layer}.hook_resid_post"
    which is the standard for GPT-2-small resid_post SAEs.

    Also sanity-checks that the SAE has enough features to cover
    the max PRISM 'unit' index for this layer.
    """
    if layer in sae_cache:
        return sae_cache[layer]

    sae_id = f"blocks.{layer}.hook_resid_post"
    print(f"\nLoading SAE for layer {layer}: release={SAE_RELEASE}, sae_id={sae_id}")

    sae, cfg, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
        SAE_RELEASE,
        sae_id,
        device=device,
    )
    sae.eval()

    # Sanity check: SAE feature width must exceed max PRISM unit for this layer
    if layer in max_unit_by_layer:
        max_unit = max_unit_by_layer[layer]

        W0, W1 = sae.W_dec.shape  # two dims
        d_model = eval_model.cfg.d_model  # 768 for gpt2-small

        # Decide which axis is features and which is d_model
        if W0 == d_model and W1 != d_model:
            n_features = W1
        elif W1 == d_model and W0 != d_model:
            n_features = W0
        else:
            # Fallback: assume features is the larger dimension
            n_features = max(W0, W1)
            print(
                f"Warning: couldn't infer orientation of W_dec cleanly; "
                f"using n_features = max({W0}, {W1}) = {n_features}"
            )

        if max_unit >= n_features:
            raise ValueError(
                f"PRISM units go up to {max_unit} in layer {layer}, "
                f"but SAE '{SAE_RELEASE}/{sae_id}' only has {n_features} features "
                f"(W_dec.shape={sae.W_dec.shape}). "
                "This means you're using the wrong SAE release for these descriptions."
            )
        else:
            print(
                f"Layer {layer}: SAE W_dec.shape={sae.W_dec.shape}, "
                f"inferred n_features={n_features}, max PRISM unit={max_unit} -> OK."
            )

    sae_cache[layer] = sae
    return sae


def hook_name_for_layer(layer: int) -> str:
    """Return the TransformerLens hook name for this layer's resid_post."""
    return f"blocks.{layer}.hook_resid_post"


# CELL 5: LOAD TEXT GENERATOR MODEL + HELPER

gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
gen_model = AutoModelForCausalLM.from_pretrained(GEN_MODEL_NAME).to(device)
gen_model.eval()

if gen_tokenizer.pad_token_id is None:
    gen_tokenizer.pad_token_id = gen_tokenizer.eos_token_id

def generate_samples(prompt: str, n_samples: int, max_new_tokens: int = MAX_NEW_TOKENS):
    """Generate n_samples texts from the generator model given a prompt."""
    samples = []
    inputs = gen_tokenizer(prompt, return_tensors="pt").to(device)
    for _ in range(n_samples):
        with torch.no_grad():
            out = gen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=gen_tokenizer.pad_token_id,
            )
        full_text = gen_tokenizer.decode(out[0], skip_special_tokens=True)
        if full_text.startswith(prompt):
            continuation = full_text[len(prompt):].strip()
        else:
            continuation = full_text
        samples.append(continuation)
    return samples


# CELL 6: LOSS COMPUTATION HELPERS (BASELINE + INTERVENTION)

def compute_losses_baseline(texts):
    """Baseline per-sample loss with no intervention."""
    if not texts:
        return np.array([]), float("nan")

    tokens = eval_model.to_tokens(texts).to(device)
    input_ids = tokens
    labels = tokens.clone()

    with torch.no_grad():
        logits = eval_model(input_ids, return_type="logits")

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    logp = torch.log_softmax(shift_logits, dim=-1)
    nll = -logp.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    token_counts = torch.ones_like(shift_labels, dtype=torch.float32).sum(dim=1)
    per_sample_loss = nll.sum(dim=1) / token_counts

    return per_sample_loss.cpu().numpy(), float(per_sample_loss.mean().item())

Dcy64pwx.dcy64pwx
# CELL 7 (UPDATED): SAE INTERVENTION HOOK (PER-LAYER) + LOSS WITH INTERVENTION

from functools import partial

def sae_intervention_hook(
    acts: torch.Tensor,
    hook,
    sae: SAE,
    feature_indices,
    mode: str = "ablate",
    clamp_values=None,
):
    """
    acts: [batch, seq, d_model] activation at blocks.{layer}.hook_resid_post
    mode:
      - "ablate": set selected features to 0
      - "clamp": set selected features to clamp_values (same length as feature_indices)
    """
    assert mode in ("ablate", "clamp")

    bsz, seq_len, d_model = acts.shape
    acts_flat = acts.reshape(-1, d_model)  # [B*T, d_model]

    with torch.no_grad():
        feats = sae.encode(acts_flat)  # [B*T, n_features]

        if not isinstance(feature_indices, (list, tuple, np.ndarray)):
            feature_indices_list = [feature_indices]
        else:
            feature_indices_list = list(feature_indices)

        if mode == "ablate":
            feats[:, feature_indices_list] = 0.0
        elif mode == "clamp":
            assert clamp_values is not None
            assert len(clamp_values) == len(feature_indices_list)
            for idx, val in zip(feature_indices_list, clamp_values):
                feats[:, idx] = val

        W_dec = sae.W_dec.to(acts_flat.dtype).to(acts_flat.device)
        b_dec = sae.b_dec.to(acts_flat.dtype).to(acts_flat.device)  # [d_model]

        d_model = acts_flat.shape[-1]
        W0, W1 = W_dec.shape

        # Case 1: W_dec is [d_model, n_features]
        if W0 == d_model and W1 != d_model:
            # feats: [B*T, n_features], W_dec.T: [n_features, d_model]
            recon_flat = feats @ W_dec.T + b_dec  # [B*T, d_model]

        # Case 2: W_dec is [n_features, d_model]
        elif W1 == d_model and W0 != d_model:
            # feats: [B*T, n_features], W_dec: [n_features, d_model]
            recon_flat = feats @ W_dec + b_dec    # [B*T, d_model]

        else:
            raise ValueError(
                f"Unexpected W_dec shape {W_dec.shape} for d_model={d_model}. "
                "Can't infer how to decode features."
            )

    recon = recon_flat.reshape(bsz, seq_len, d_model)
    return recon


def compute_losses_with_intervention(
    texts,
    sae: SAE,
    hook_name: str,
    feature_indices,
    mode: str = "ablate",
    clamp_values=None,
):
    """Compute per-sample loss with SAE intervention at a specific layer."""
    if not texts:
        return np.array([]), float("nan")

    tokens = eval_model.to_tokens(texts).to(device)
    input_ids = tokens
    labels = tokens.clone()

    hook_fn = partial(
        sae_intervention_hook,
        sae=sae,
        feature_indices=feature_indices,
        mode=mode,
        clamp_values=clamp_values,
    )
    fwd_hooks = [(hook_name, hook_fn)]

    with torch.no_grad():
        logits = eval_model.run_with_hooks(
            input_ids,
            return_type="logits",
            fwd_hooks=fwd_hooks,
        )

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    logp = torch.log_softmax(shift_logits, dim=-1)
    nll = -logp.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    token_counts = torch.ones_like(shift_labels, dtype=torch.float32).sum(dim=1)
    per_sample_loss = nll.sum(dim=1) / token_counts

    return per_sample_loss.cpu().numpy(), float(per_sample_loss.mean().item())


# CELL 8: PROMPT TEMPLATES + GENERATION

def build_single_concept_prompt(desc: str) -> str:
    return (
        "Write a short paragraph (4–6 sentences) that strongly involves the following theme:\n"
        f"\"{desc}\"\n"
        "Focus ONLY on this theme.\n"
        "Avoid mentioning or alluding to unrelated topics.\n\n"
    )

def build_pair_concept_prompt(desc1: str, desc2: str) -> str:
    return (
        "Write a short paragraph (6–8 sentences) that strongly involves BOTH of the following themes:\n"
        f"1. \"{desc1}\"\n"
        f"2. \"{desc2}\"\n\n"
        "Make sure both themes appear multiple times and interact in a coherent way.\n"
        "Do not write a list; write a natural paragraph.\n\n"
    )

all_feature_entries = []

for (layer, unit), df_feat in feature_concepts.items():
    print(f"\n=== Generating data for feature (layer={layer}, unit={unit}) ===")
    descriptions = df_feat["description"].tolist()
    n_desc = len(descriptions)
    print("Descriptions:")
    for i, d in enumerate(descriptions):
        print(f"  concept[{i}]: {d}")

    single_texts = {}  # concept index -> list of texts
    pair_texts = {}    # (i, j) -> list of texts

    # Single-concept texts
    for i, desc in enumerate(descriptions):
        prompt = build_single_concept_prompt(desc)
        texts_i = generate_samples(prompt, n_samples=N_SAMPLES_PER_CONCEPT)
        single_texts[i] = texts_i

    # Pair-concept texts (all unordered pairs)
    for i, j in itertools.combinations(range(n_desc), 2):
        desc_i, desc_j = descriptions[i], descriptions[j]
        prompt_pair = build_pair_concept_prompt(desc_i, desc_j)
        texts_ij = generate_samples(prompt_pair, n_samples=N_SAMPLES_PER_PAIR)
        pair_texts[(i, j)] = texts_ij

    all_feature_entries.append(
        {
            "layer": layer,
            "unit": unit,
            "df_feat": df_feat,
            "descriptions": descriptions,
            "single_texts": single_texts,
            "pair_texts": pair_texts,
        }
    )

print("\nFinished generating prompts for all selected features.")


# CELL 9: RUN BASELINE + ABLATION LOSSES FOR EACH FEATURE

results_rows = []

for feat_entry in all_feature_entries:
    layer = feat_entry["layer"]
    unit = feat_entry["unit"]
    df_feat = feat_entry["df_feat"]
    descriptions = feat_entry["descriptions"]
    single_texts = feat_entry["single_texts"]
    pair_texts = feat_entry["pair_texts"]

    # Load correct SAE and hook_name for this layer
    sae = get_sae_for_layer(layer)
    hook_name = hook_name_for_layer(layer)

    feature_idx = unit  # PRISM's 'unit' is the SAE feature index for this layer

    print(f"\n=== Evaluating feature (layer={layer}, unit={unit}) ===")

    # SINGLE-CONCEPT PROMPTS
    for concept_id, texts in single_texts.items():
        concept_label = f"C{concept_id}"

        # Baseline
        losses_base, _ = compute_losses_baseline(texts)

        # Ablated (feature_idx)
        losses_abl, _ = compute_losses_with_intervention(
            texts,
            sae=sae,
            hook_name=hook_name,
            feature_indices=[feature_idx],
            mode="ablate",
            clamp_values=None,
        )

        for i, (text, lb, la) in enumerate(zip(texts, losses_base, losses_abl)):
            results_rows.append(
                {
                    "layer": layer,
                    "unit": unit,
                    "concept_set": concept_label,
                    "sample_type": "single",
                    "sample_idx": i,
                    "mode": "baseline",
                    "loss": float(lb),
                    "text": text,
                }
            )
            results_rows.append(
                {
                    "layer": layer,
                    "unit": unit,
                    "concept_set": concept_label,
                    "sample_type": "single",
                    "sample_idx": i,
                    "mode": "ablate",
                    "loss": float(la),
                    "text": text,
                }
            )

    # PAIR-CONCEPT PROMPTS
    for (i, j), texts in pair_texts.items():
        concept_label = f"C{i}+C{j}"

        losses_base, _ = compute_losses_baseline(texts)
        losses_abl, _ = compute_losses_with_intervention(
            texts,
            sae=sae,
            hook_name=hook_name,
            feature_indices=[feature_idx],
            mode="ablate",
            clamp_values=None,
        )

        for k, (text, lb, la) in enumerate(zip(texts, losses_base, losses_abl)):
            results_rows.append(
                {
                    "layer": layer,
                    "unit": unit,
                    "concept_set": concept_label,
                    "sample_type": "pair",
                    "sample_idx": k,
                    "mode": "baseline",
                    "loss": float(lb),
                    "text": text,
                }
            )
            results_rows.append(
                {
                    "layer": layer,
                    "unit": unit,
                    "concept_set": concept_label,
                    "sample_type": "pair",
                    "sample_idx": k,
                    "mode": "ablate",
                    "loss": float(la),
                    "text": text,
                }
            )

print("\nFinished running losses for all features and prompts.")


# CELL 10: ASSEMBLE RESULTS AND SAVE

results_df = pd.DataFrame(results_rows)
print("Results head:")
print(results_df.head())



# COMPUTE PER-SAMPLE DELTA LOSS (ABLATE - BASELINE)

# Pivot modes so we can compute delta per sample
pivot_cols = ["layer", "unit", "concept_set", "sample_type", "sample_idx"]
pivot_df = results_df.pivot_table(
    index=pivot_cols,
    columns="mode",
    values="loss"
).reset_index()

# Compute delta = ablate - baseline
pivot_df["delta"] = pivot_df["ablate"] - pivot_df["baseline"]

# PER-FEATURE SUMMARY (SINGLE vs PAIR)

group_cols = ["layer", "unit", "sample_type", "concept_set"]

agg_df = pivot_df.groupby(group_cols).agg(
    mean_baseline=("baseline", "mean"),
    mean_ablate=("ablate", "mean"),
    mean_delta=("delta", "mean"),
    n_samples=("delta", "size"),
).reset_index()

print("Per (feature, concept_set) summary:")
print(agg_df.head())


# CELL D: COLLISION METRIC PER FEATURE

# Separate singles and pairs
singles = agg_df[agg_df["sample_type"] == "single"]
pairs   = agg_df[agg_df["sample_type"] == "pair"]

# mean over concept_sets for each feature
single_feat = singles.groupby(["layer", "unit"]).agg(
    delta_single_mean=("mean_delta", "mean"),
    delta_single_std=("mean_delta", "std"),
    n_single_sets=("mean_delta", "size"),
).reset_index()

pair_feat = pairs.groupby(["layer", "unit"]).agg(
    delta_pair_mean=("mean_delta", "mean"),
    delta_pair_std=("mean_delta", "std"),
    n_pair_sets=("mean_delta", "size"),
).reset_index()

# merge
feat_summary = single_feat.merge(
    pair_feat,
    on=["layer", "unit"],
    how="outer",
    suffixes=("_single", "_pair")
)

# Compute collision penalty where both exist
feat_summary["collision_penalty"] = (
    feat_summary["delta_pair_mean"] - feat_summary["delta_single_mean"]
)


# CELL E: JOIN WITH PRISM METRICS

# Keep unique per-feature metrics from merged_df
metrics_cols = ["layer", "unit", "cosine_similarity", "cosine_similarity_random", "max_auc", "max_mad"]
metrics_unique = merged_df[metrics_cols].drop_duplicates(subset=["layer", "unit"])

feat_with_metrics = feat_summary.merge(
    metrics_unique,
    on=["layer", "unit"],
    how="left"
)

print("Feature-level dataframe with collision + PRISM metrics:")
print(feat_with_metrics.head())

# Example: look at top features by collision_penalty
print("\nTop 10 features by collision_penalty:")
print(
    feat_with_metrics.sort_values("collision_penalty", ascending=False)
                     .head(10)[["layer", "unit", "collision_penalty", "delta_single_mean", "delta_pair_mean", "cosine_similarity", "max_auc", "max_mad"]]
)



# 1) Join the text back into pivot_df
pivot_with_text = pivot_df.merge(
    results_df[["layer", "unit", "concept_set", "sample_type", "sample_idx", "text"]].drop_duplicates(),
    on=["layer", "unit", "concept_set", "sample_type", "sample_idx"],
    how="left"
)


print("Final dataset size:", len(final_prompt_dataset))
print(final_prompt_dataset["is_runtime_collision"].value_counts())
# 2) Decide collision score = delta
pivot_with_text["collision_score"] = pivot_with_text["delta"]

# 3) Choose a threshold for runtime collision
# You can set this manually or compute from statistics.

pivot_with_text["is_runtime_collision"] = (
    pivot_with_text["sample_type"] == "pair"
).astype(int)


# 4) Final dataset of prompts
final_prompt_dataset = pivot_with_text[
    ["text", "collision_score", "is_runtime_collision",
     "layer", "unit", "concept_set", "sample_type", "sample_idx"]
].copy()

# 5) Save the dataset to disk
out_path_csv = os.path.join(OUTPUT_DIR, f"runtime_collision_prompts_layer{features_to_test[0][0]}.csv")
out_path_pkl = os.path.join(OUTPUT_DIR, f"runtime_collision_prompts_layer{features_to_test[0][0]}.pkl")

final_prompt_dataset.to_csv(out_path_csv, index=False)
final_prompt_dataset.to_pickle(out_path_pkl)

print(f"\nSaved runtime collision prompt dataset:")
print(f"CSV : {out_path_csv}")
print(f"PKL : {out_path_pkl}")