import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformer_lens import HookedTransformer
from sae_lens import SAE
from src.download_data import load_finqa_from_disk
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")
login(hf_token)



train_data = load_finqa_from_disk("FinQA/dataset/train.json")


sample = train_data.shuffle(seed=42)[0]
if isinstance(sample["table"], list):
    # Join rows nicely
    table_text = "\n".join([" | ".join(map(str, row)) for row in sample["table"]])
else:
    table_text = str(sample["table"])

context = f"{sample['pre_text']}\n{table_text}\n{sample['post_text']}"
prompt = f"Context:\n{context}\n\nQuestion: {sample['question']}\nAnswer:"


MODEL_NAME = "meta-llama/Llama-3.2-1B"
TARGET_LAYER = 12
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)

sae_release = "seonglae/Llama-3.2-1B-sae"
sae_id = "Llama-3.2-1B_blocks.12.hook_resid_pre_14336_topk_48_0.0002_49_fineweb_512"
sae, cfg_dict, sparsity = SAE.from_pretrained(sae_release, sae_id, device=device)


# tokenize and get activations
tokens = model.to_tokens(prompt)
logits, cache = model.run_with_cache(tokens, remove_batch_dim=False, return_type="logits")
acts = cache["resid_pre", TARGET_LAYER]


feature_acts = sae.encode(acts)
feature_mean = feature_acts.mean().item()
feature_std = feature_acts.std().item()


topk_values, topk_indices = torch.topk(feature_acts.mean(dim=0), k=10)

for val, idx in zip(topk_values, topk_indices):
    print(f"Feature {idx.item():>6} | Activation {val.item():.4f}")


log_probs = F.log_softmax(logits, dim=-1)

target_tokens = tokens[:, 1:]
pred_log_probs = log_probs[:, :-1, :]

token_log_probs = pred_log_probs.gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)

nll_per_token = -token_log_probs