print("Starting main.py")
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
import json
import numpy as np
import pandas as pd
import torch
import pickle
import lzma
import joblib
import os
import gc
import time


from sklearn.model_selection import train_test_split, GridSearchCV
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

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
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
MODEL_NAME = "google/gemma-2-2b"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"


# MODEL_NAME = 'gpt2-small'
# SAE_RELEASE = "gpt2-small-resid-post-v5-32k"
#SAE_ID = 'blocks.0.hook_resid_post'
print("Using model:", MODEL_NAME)
print("Using SAE release:", SAE_RELEASE)
model = HookedTransformer.from_pretrained_no_processing(
model_name=MODEL_NAME, device=device, dtype=DTYPE
)
model.eval()
TARGET_LAYER = 10
CLUSTER_COUNT = 10
print("Target layer for analysis:", TARGET_LAYER)

TOPK_PER_FEATURE = 250
CANDIDATE_FEATURES_FOR_ENTROPY = 800
BATCH_SIZE = 16

SAE_ID = f"layer_{TARGET_LAYER}/width_16k/canonical"
SAE_ID_SAVE = f"layer_{TARGET_LAYER}_16_k_gemma"
PRED_OUT = f"results/runtime_collision_preds_with_proba_{SAE_ID_SAVE}.csv"



os.makedirs("outputs", exist_ok=True)
data = pl.read_csv(f"data/gemma_data/gemma_layer{TARGET_LAYER}.csv")
data = data.to_pandas()
N_SAMPLES = len(data)


# sample N_SAMPLES rows
sample_rows = data.sample(n=N_SAMPLES).reset_index(drop=True)

prompts = [s for s in sample_rows["text"]]

sample_rows.columns


# load env variables
login()

sae, cfg_dict, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
    SAE_RELEASE,
    SAE_ID,
    device=device,
)
sae = sae.to(dtype=DTYPE)

# tokenize
pad_id = model.tokenizer.pad_token_id
if pad_id is None:
    pad_id = model.tokenizer.eos_token_id


per_prompt_loss_base = []
per_prompt_spike_90 = []

all_acts_flat = []
all_tokens_flat = []
all_tokens_2d = []
prompt_lengths = []


for i in range(0, len(prompts), BATCH_SIZE):
    batch_prompts = prompts[i:i+BATCH_SIZE]

    tokens = model.to_tokens(batch_prompts).to(device)
    labels = tokens.clone()

    pad_id = model.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = model.tokenizer.eos_token_id
    lengths = (labels != pad_id).sum(dim=1).cpu().tolist()
    prompt_lengths.extend(lengths)

    mask = labels[:, 1:] != pad_id

    with torch.no_grad():
        logits, cache = model.run_with_cache(
            tokens,
            return_type="logits",
            names_filter=[f"blocks.{TARGET_LAYER}.hook_resid_post"],
        )

    acts = cache[f"blocks.{TARGET_LAYER}.hook_resid_post"].detach()
    del cache

    with torch.no_grad():
        feats = sae.encode(acts)

    # loss computation
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    logp = F.log_softmax(shift_logits, dim=-1)
    nll = -logp.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    nll = nll * mask.float()

    loss_per_prompt = nll.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    spike_90 = torch.nanquantile(
        nll.masked_fill(~mask, float("nan")), 0.9, dim=1
    )

    per_prompt_loss_base.append(loss_per_prompt.cpu())
    per_prompt_spike_90.append(spike_90.cpu())

    flat_mask = (tokens.reshape(-1) != pad_id).cpu()

    feats_flat = feats.reshape(-1, feats.size(-1)).detach().cpu()
    tokens_flat = tokens.reshape(-1).detach().cpu()

    feats_flat = feats_flat[flat_mask]
    tokens_flat = tokens_flat[flat_mask]

    all_acts_flat.append(feats_flat)
    all_tokens_flat.append(tokens_flat)




    del logits, acts, feats, tokens, labels
    torch.cuda.empty_cache()


# concatenate prompts
per_prompt_loss_base = torch.cat(per_prompt_loss_base).numpy()
per_prompt_spike_90 = torch.cat(per_prompt_spike_90).numpy()
feature_acts_flat = torch.cat(all_acts_flat, dim=0)
tokens_flat = torch.cat(all_tokens_flat, dim=0)


torch.save(tokens_flat, f"outputs/tokens_{SAE_ID_SAVE}.pt")
tokens_flat = torch.cat(all_tokens_flat, dim=0).numpy()

prompt_lengths = np.array(prompt_lengths, dtype=np.int32)
np.save(f"outputs/prompt_lengths_{SAE_ID_SAVE}.npy", prompt_lengths)

mean_nll = per_prompt_loss_base.mean()
ppl_base = float(np.exp(mean_nll))

print(f"Overall perplexity: {ppl_base:.4f}")

MAX_TOKENS = 2_000_000
if feature_acts_flat.size(0) > MAX_TOKENS:
    idx = torch.randperm(feature_acts_flat.size(0))[:MAX_TOKENS]
    feature_acts_flat = feature_acts_flat[idx]
    tokens_flat = tokens_flat[idx]
else:
    idx = torch.arange(feature_acts_flat.size(0))
np.save(f"outputs/token_subsample_idx_{SAE_ID_SAVE}.npy", idx.cpu().numpy())


W_dec = sae.W_dec.detach().cpu().T
poly_score_decoder = torch.sum(W_dec.abs(), dim=0)

cand_idx = torch.topk(
    poly_score_decoder,
    k=min(CANDIDATE_FEATURES_FOR_ENTROPY, poly_score_decoder.numel()),
    largest=True,
).indices.cpu()

np.save(
    f"outputs/cand_idx_{SAE_ID_SAVE}.npy",
    cand_idx.detach().cpu().numpy()
)
feature_acts_cand = feature_acts_flat[:, cand_idx]

token_embeddings = model.W_E.detach().cpu().to(torch.float32).numpy()



cluster_results = {}
print("Clustering top features...")
def compute_clusters_for_feature(j, feature_id, top_positions_cand):
    pos_idx = top_positions_cand[:, j]
    ids = tokens_flat[pos_idx]
    X = token_embeddings[ids]

    clustering = AgglomerativeClustering(
        n_clusters=CLUSTER_COUNT,
        linkage="average",
        metric="cosine",
    )
    clustering.fit(X)

    # dcode token strings
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
top_positions_cand = torch.topk(
    feature_acts_cand,
    k=min(TOPK_PER_FEATURE, feature_acts_cand.size(0)),
    dim=0
).indices.cpu().numpy()

np.save(f"outputs/top_positions_{SAE_ID_SAVE}.npy", top_positions_cand)


for j, f in enumerate(cand_idx.tolist()):
    cluster_results[f] = compute_clusters_for_feature(j, f, top_positions_cand)

with lzma.open(f"outputs/feature_clusters_{SAE_ID_SAVE}.pkl.xz", "wb") as f:
    pickle.dump(cluster_results, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saved clusters for {len(cluster_results)} features to outputs/feature_clusters_{SAE_ID_SAVE}.pkl.xz")

torch.save(feature_acts_cand, f"outputs/feature_acts_cand_{SAE_ID_SAVE}.pt")

print("Computing per-prompt metrics...")

per_prompt_ppl_base = np.exp(per_prompt_loss_base)
ppl_base = float(np.exp(per_prompt_loss_base.mean()))

per_prompt_ppl_base = np.exp(per_prompt_loss_base)

df_training = pd.DataFrame({
    "unit": sample_rows["unit"].values,
    "prompt": prompts,
    "ppl_base": per_prompt_ppl_base,
    "next_token_loss_base": per_prompt_loss_base,
    "original_index": sample_rows.index.values,
    "ppl_spike_90": per_prompt_spike_90,
    "label": sample_rows["label"].values,
})
df_training.to_csv(f"outputs/training_dataset_{SAE_ID_SAVE}.csv", index=False)