import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformer_lens import HookedTransformer
from sae_lens import SAE
from sae_lens import list_pretrained_saes

for sae in list_pretrained_saes():
    print(sae)

device = "cuda" if torch.cuda.is_available() else "cpu"

hf_token = os.getenv("HUGGINS_FACE_TOKEN")

login(hf_token)
## requires manual download, first time model is used
## how to delete again find folder and delete, C:\Users\<you>\.cache\huggingface\hub\models--TinyLlama--TinyLlama-1.1B-Chat-v1.0
## according to the chat
TINY_TEST_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
## has to set up hugging face account and accept terms of use, and request model access
# to use big boy model
MODEL_NAME = "meta-llama/Llama-3.2-1B"

TARGET_LAYER = 12 

## setup
device = "cuda" if torch.cuda.is_available() else "cpu"

model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
print(sae.cfg)  # or sae.cfg.__dict__


# only certain pretrained SAEs are available, and they have to match layer and dum stuff in weird ways,
# i suppose this works now maybe, i geuss?????
### omg this takes forever to load, fuck off
# apparently this works only with layer 12, i cant be botherd to find sae that works on later layers
sae, cfg_dict, sparsity = SAE.from_pretrained("seonglae/Llama-3.2-1B-sae", "Llama-3.2-1B_blocks.12.hook_resid_pre_14336_topk_48_0.0002_49_fineweb_512")

prompt = "The financial markets crashed because investors panicked."
tokens = model.to_tokens(prompt)
_, cache = model.run_with_cache(tokens, remove_batch_dim=True)

# getting the activation from the specified layer, 12
acts = cache["resid_pre", TARGET_LAYER]  


feature_acts = sae.encode(acts)
feature_mean = feature_acts.mean().item()
feature_std = feature_acts.std().item()
