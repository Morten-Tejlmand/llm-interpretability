import os
import random
import torch
import torch.nn.functional as F
import pandas as pd
import polars as pl
import altair as alt
from dotenv import load_dotenv
from huggingface_hub import login
from transformer_lens import HookedTransformer
from sae_lens import SAE
import numpy as np
from torch.nn.functional import log_softmax, softmax, cosine_similarity

# CONFIG
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "meta-llama/Llama-3.2-1B"
SAE_RELEASE = "seonglae/Llama-3.2-1B-sae"
SAE_ID = "Llama-3.2-1B_blocks.12.hook_resid_pre_14336_topk_48_0.0002_49_fineweb_512"
TARGET_LAYER = 12
N_SAMPLES = 1

# DATA
df_wiki = pl.read_csv("wikipedia_data.csv")
df_wiki = df_wiki.filter(pl.col("text").str.len_chars() > 10)
samples = [df_wiki[i]["text"] for i in random.sample(range(len(df_wiki)), N_SAMPLES)]

prompts = [f"Context:\n{s}\n\nQuestion: Summarize the above text.\nAnswer:" for s in samples]

# MODEL & SAE
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")
login(hf_token)

model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=torch.float16)
sae, cfg_dict, sparsity = SAE.from_pretrained(SAE_RELEASE, SAE_ID, device=device)

tokens = model.to_tokens(prompts)
labels = tokens.clone()

# BASELINE FORWARD (no SAE)
with torch.no_grad():
    logits, cache = model.run_with_cache(tokens, remove_batch_dim=False, return_type="logits")

shift_logits = logits[:, :-1, :].contiguous()
shift_labels = labels[:, 1:].contiguous()
loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="mean")
ppl_base = torch.exp(loss).item()
print(f"Baseline Perplexity: {ppl_base:.2f}")

# GET SAE FEATURES
acts = cache["resid_pre", TARGET_LAYER]
feature_acts = sae.encode(acts)
feature_means = feature_acts.mean(dim=(0, 1))

# POLYSEMANTICITY: Feature–Feature correlation
*# props not done correctly, dont know how to
flat_feats = torch.nn.functional.normalize(feature_acts.mean(dim=1), dim=1)
sim = (flat_feats.T @ flat_feats / flat_feats.shape[0]).to(torch.float32)

polysemanticity = torch.mean(torch.abs(sim), dim=1)
polysemanticity_np = polysemanticity.cpu().detach().numpy()

sorted_idx = np.argsort(polysemanticity_np)[::-1]

top_poly_idx = torch.tensor(sorted_idx[:50].copy(), dtype=torch.long, device=device)
low_poly_idx = torch.tensor(sorted_idx[-50:].copy(), dtype=torch.long, device=device)

# BASELINE NEXT-TOKEN LOSS & PROBABILITIE
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = labels[:, 1:].contiguous()

loss_base = F.cross_entropy(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1),
    reduction="mean"
)
ppl_base = torch.exp(loss_base).item()

p_base = softmax(shift_logits, dim=-1)

# first intervention: Zero-out high-polysemantic features
def hook_remove_poly(value, hook):
    z = sae.encode(value)
    z[:, :, top_poly_idx] = 0
    recon = sae.decode(z)
    return recon

with torch.no_grad():
    logits_mod = model.run_with_hooks(
        tokens,
        fwd_hooks=[(f"blocks.{TARGET_LAYER}.hook_resid_pre", hook_remove_poly)],
        return_type="logits",
    )

shift_logits_mod = logits_mod[:, :-1, :].contiguous()
loss_mod = F.cross_entropy(
    shift_logits_mod.view(-1, shift_logits_mod.size(-1)),
    shift_labels.view(-1),
    reduction="mean"
)
ppl_mod = torch.exp(loss_mod).item()
p_mod = softmax(shift_logits_mod, dim=-1)

# second option: Zero-out least-polysemantic features
def hook_remove_mono(value, hook):
    z = sae.encode(value)
    z[:, :, low_poly_idx] = 0
    recon = sae.decode(z)
    return recon

with torch.no_grad():
    logits_mod2 = model.run_with_hooks(
        tokens,
        fwd_hooks=[(f"blocks.{TARGET_LAYER}.hook_resid_pre", hook_remove_mono)],
        return_type="logits",
    )

shift_logits_mod2 = logits_mod2[:, :-1, :].contiguous()
loss_mod2 = F.cross_entropy(
    shift_logits_mod2.view(-1, shift_logits_mod2.size(-1)),
    shift_labels.view(-1),
    reduction="mean"
)
ppl_mod2 = torch.exp(loss_mod2).item()
p_mod2 = softmax(shift_logits_mod2, dim=-1)

# EXTRA METRICS

def mean_kl_divergence(p_ref, p_alt):
    """Average KL divergence D_KL(p_ref || p_alt) across batch and sequence."""
    p_ref = p_ref.float()
    p_alt = p_alt.float()
    log_p_ref = torch.log(torch.clamp(p_ref, 1e-9, 1.0))
    log_p_alt = torch.log(torch.clamp(p_alt, 1e-9, 1.0))
    kl = torch.sum(p_ref * (log_p_ref - log_p_alt), dim=-1)
    return kl.mean().item()

def mean_cosine_similarity(logits_ref, logits_alt):
    """Average cosine similarity between token-level logits."""
    sims = cosine_similarity(
        logits_ref.view(-1, logits_ref.size(-1)),
        logits_alt.view(-1, logits_alt.size(-1)),
        dim=-1,
    )
    return sims.mean().item()

kl_poly = mean_kl_divergence(p_base, p_mod)
kl_mono = mean_kl_divergence(p_base, p_mod2)

cos_poly = mean_cosine_similarity(shift_logits, shift_logits_mod)
cos_mono = mean_cosine_similarity(shift_logits, shift_logits_mod2)


df_summary = pd.DataFrame({
    "Condition": ["Baseline", "Remove Polysemantic", "Remove Monosemantic"],
    "Next-token Loss": [loss_base.item(), loss_mod.item(), loss_mod2.item()],
    "Perplexity": [ppl_base, ppl_mod, ppl_mod2],
    "Δ Perplexity": [0, ppl_mod - ppl_base, ppl_mod2 - ppl_base],
    "Mean KL (vs Base)": [0, kl_poly, kl_mono],
    "Cosine Sim (vs Base)": [1.0, cos_poly, cos_mono]
})

print(df_summary)

chart = (
    alt.Chart(df_summary)
    .mark_bar()
    .encode(
        x="Condition:N",
        y="Perplexity:Q",
        color="Condition:N",
        tooltip=[
            "Next-token Loss",
            "Perplexity",
            "Δ Perplexity",
            "Mean KL (vs Base)",
            "Cosine Sim (vs Base)",
        ],
    )
    .properties(width=550, height=300, title="Effect of Feature Removal on Output Metrics")
)
chart.display()