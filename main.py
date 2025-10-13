import os
import random
import torch
import torch.nn.functional as F
import pandas as pd
import altair as alt
from dotenv import load_dotenv
from huggingface_hub import login
from transformer_lens import HookedTransformer
from sae_lens import SAE
from src.download_data import load_finqa_from_disk


device = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 1
MODEL_NAME = "meta-llama/Llama-3.2-1B"
TARGET_LAYER = 12
SAE_RELEASE = "seonglae/Llama-3.2-1B-sae"
SAE_ID = "Llama-3.2-1B_blocks.12.hook_resid_pre_14336_topk_48_0.0002_49_fineweb_512"

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")
login(hf_token)


# finance prompts
# ----------
train_data = load_finqa_from_disk("FinQA/dataset/train.json")
samples = [train_data[i] for i in random.sample(range(len(train_data)), N_SAMPLES)]

prompts = []
for s in samples:
    table_text = "\n".join([" | ".join(map(str, row)) for row in s["table"]]) if isinstance(s["table"], list) else str(s["table"])
    context = f"{s['pre_text']}\n{table_text}\n{s['post_text']}"
    prompt = f"Context:\n{context}\n\nQuestion: {s['question']}\nAnswer:"
    prompts.append(prompt)

# model stuff
# ----------------
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=torch.float32)
sae, cfg_dict, sparsity = SAE.from_pretrained(SAE_RELEASE, SAE_ID, device=device)

tokens = model.to_tokens(prompts)
logits, cache = model.run_with_cache(tokens, remove_batch_dim=False, return_type="logits")

acts = cache["resid_pre", TARGET_LAYER]  # [batch, seq, d_model]

# sae encoding and feature analysis 
# -----------------
feature_acts = sae.encode(acts)  # [batch, seq, n_features]
feature_means = feature_acts.mean(dim=(0, 1))
topk_values, topk_indices = torch.topk(feature_means, k=20)

# looking at some cosine similarities, to inspect polysemantic neurons
# ---------------------------
 
W_dec = sae.W_dec.detach()
neuron_basis = torch.eye(W_dec.shape[1], device=W_dec.device)

similarity = F.cosine_similarity(
    W_dec.unsqueeze(1), neuron_basis.unsqueeze(0), dim=-1
)

top_neuron_sim, top_neuron_idx = similarity.max(dim=1)



# top features plot
# ---------------------------
df_top_features = pd.DataFrame({
    "Feature ID": topk_indices.detach().cpu().to(torch.int32).numpy(),
    "Mean Activation": topk_values.detach().cpu().to(torch.float32).numpy()
})

chart_top_features = (
    alt.Chart(df_top_features)
    .mark_bar()
    .encode(
        x=alt.X("Feature ID:O", sort="-y"),
        y="Mean Activation:Q",
        tooltip=["Feature ID", "Mean Activation"]
    )
    .properties(width=600, height=300, title="Top 20 SAE Features by Mean Activation")
)
chart_top_features.display()

BATCH_IDX = 0
FEATURE_ID = int(topk_indices[0].item())
tokens_str = model.to_str_tokens(prompts[BATCH_IDX])
feature_map = feature_acts[BATCH_IDX, :, FEATURE_ID].detach().cpu().to(torch.float32).numpy()

df_tokens = pd.DataFrame({
    "Token": tokens_str,
    "Activation": feature_map,
    "Position": list(range(len(tokens_str)))
})

chart_heatmap = (
    alt.Chart(df_tokens)
    .mark_rect()
    .encode(
        x=alt.X("Position:O", axis=alt.Axis(labelAngle=0)),
        color=alt.Color("Activation:Q", scale=alt.Scale(scheme="viridis")),
        tooltip=["Token", "Activation"]
    )
    .properties(width=800, height=100, title=f"Feature {FEATURE_ID} Activation Across Tokens")
)
chart_heatmap.display()

# create some plots and stuff
# ---------------------------
flat_feats = feature_acts.mean(dim=1)
flat_feats = F.normalize(flat_feats, dim=1)
sim = flat_feats.T @ flat_feats / flat_feats.shape[0]
collision_matrix = sim.detach().cpu().to(torch.float32).numpy()

rows, cols = collision_matrix.shape
df_corr = pd.DataFrame(
    [(i, j, collision_matrix[i, j]) for i in range(rows) for j in range(cols)],
    columns=["Feature A", "Feature B", "Correlation"]
)

chart_corr = (
    alt.Chart(df_corr)
    .mark_rect()
    .encode(
        x=alt.X("Feature A:O", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Feature B:O"),
        color=alt.Color("Correlation:Q", scale=alt.Scale(scheme="redblue", domain=(-1, 1))),
        tooltip=["Feature A", "Feature B", "Correlation"]
    )
    .properties(width=400, height=400, title="Feature-Feature Correlation Matrix")
)
chart_corr.display()

def get_feature_top_tokens(feature_id, k=20):
    flat = feature_acts[:, :, feature_id].flatten()
    topk = torch.topk(flat, k)
    indices = topk.indices
    data = []
    for idx in indices:
        b, t = divmod(idx.item(), feature_acts.shape[1])
        token_id = tokens[b, t].item()
        token = model.to_single_str_token(token_id)
        data.append({"Feature": feature_id, "Token": token, "Activation": flat[idx].item()})
    return pd.DataFrame(data)

df_tok = pd.concat([get_feature_top_tokens(int(fid), 15) for fid in topk_indices[:10]])
chart_tokens = (
    alt.Chart(df_tok)
    .mark_bar()
    .encode(
        x=alt.X("Activation:Q", title="Activation Strength"),
        y=alt.Y("Token:N", sort="-x", title="Token"),
        color="Feature:N",
        tooltip=["Feature", "Token", "Activation"]
    )
    .properties(width=600, height=200)
    .facet(
        row=alt.Row("Feature:N",
                    header=alt.Header(labelFontSize=14, title="Feature"))
    )
    .interactive(bind_x=True, bind_y=False)
    .properties(title="Top Activating Tokens per Feature (Zoom & Pan)")
)

chart_tokens.display()
