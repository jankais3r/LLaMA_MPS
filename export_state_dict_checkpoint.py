#!/usr/bin/env python3
# modified from https://github.com/tloen/alpaca-lora/blob/main/export_state_dict_checkpoint.py

from transformers import LlamaTokenizer, LlamaForCausalLM
import os
import sys
import json
import torch
import transformers
from peft import PeftModel, LoraConfig

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip3 uninstall transformers && pip3 install git+https://github.com/huggingface/transformers.git"


def info_and_exit():
    print("Run as: python3 export_state_dict_checkpoint.py 7B")
    print("     or python3 export_state_dict_checkpoint.py 13B")
    print("     or python3 export_state_dict_checkpoint.py 30B")
    sys.exit()


if len(sys.argv) != 2:
    info_and_exit()

models_info = {
    "7B": [
        "tloen/alpaca-lora-7b",
        {"dim": 4096, "multiple_of": 256, "n_heads": 32,
            "n_layers": 32, "norm_eps": 1e-06, "vocab_size": -1, }
    ],
    "13B": [
        "samwit/alpaca13B-lora",
        {"dim": 5120, "multiple_of": 256, "n_heads": 40,
         "n_layers": 40, "norm_eps": 1e-06, "vocab_size": -1, }
    ],
    "30B": [
        "baseten/alpaca-30b",
        {"dim": 6656, "multiple_of": 256, "n_heads": 52,
            "n_layers": 60, "norm_eps": 1e-06, "vocab_size": -1, }
    ],
}


def download_model(n):
    d_llama = f"decapoda-research/llama-{n}b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(d_llama)
    base_model = LlamaForCausalLM.from_pretrained(
        d_llama,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )
    lora_model = PeftModel.from_pretrained(
        base_model,
        models_info[f"{n}B"][0],
        device_map={"": "cpu"},
        torch_dtype=torch.float16,
    )
    params = models_info[f"{n}B"][1]
    return tokenizer, base_model, lora_model, params


# get n from 7B, 13B, 30B
n = int(sys.argv[1][:-1])
if n in [7, 13, 30]:
    tokenizer, base_model, lora_model, params = download_model(n)
else:
    info_and_exit()

for layer in lora_model.base_model.model.model.layers:
    layer.self_attn.q_proj.merge_weights = True
    layer.self_attn.v_proj.merge_weights = True

lora_model.train(False)
lora_model_sd = lora_model.state_dict()
n_layers = params["n_layers"]
n_heads = params["n_heads"]
dim = params["dim"]
dims_per_head = dim // n_heads
base = 10000.0
inv_freq = 1.0 / \
    (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))


def permute(w):
    return (
        w.view(n_heads, dim // n_heads // 2, 2,
               dim).transpose(1, 2).reshape(dim, dim)
    )


def unpermute(w):
    return (
        w.view(n_heads, 2, dim // n_heads // 2,
               dim).transpose(1, 2).reshape(dim, dim)
    )


def translate_state_dict_key(k):
    k = k.replace("base_model.model.", "")
    if k == "model.embed_tokens.weight":
        return "tok_embeddings.weight"
    elif k == "model.norm.weight":
        return "norm.weight"
    elif k == "lm_head.weight":
        return "output.weight"
    elif k.startswith("model.layers."):
        layer = k.split(".")[2]
        if k.endswith(".self_attn.q_proj.weight"):
            return f"layers.{layer}.attention.wq.weight"
        elif k.endswith(".self_attn.k_proj.weight"):
            return f"layers.{layer}.attention.wk.weight"
        elif k.endswith(".self_attn.v_proj.weight"):
            return f"layers.{layer}.attention.wv.weight"
        elif k.endswith(".self_attn.o_proj.weight"):
            return f"layers.{layer}.attention.wo.weight"
        elif k.endswith(".mlp.gate_proj.weight"):
            return f"layers.{layer}.feed_forward.w1.weight"
        elif k.endswith(".mlp.down_proj.weight"):
            return f"layers.{layer}.feed_forward.w2.weight"
        elif k.endswith(".mlp.up_proj.weight"):
            return f"layers.{layer}.feed_forward.w3.weight"
        elif k.endswith(".input_layernorm.weight"):
            return f"layers.{layer}.attention_norm.weight"
        elif k.endswith(".post_attention_layernorm.weight"):
            return f"layers.{layer}.ffn_norm.weight"
        elif k.endswith("rotary_emb.inv_freq") or "lora" in k:
            return None
        else:
            print(layer, k)
            raise NotImplementedError
    else:
        print(k)
        raise NotImplementedError


new_state_dict = {}
for k, v in lora_model_sd.items():
    new_k = translate_state_dict_key(k)
    if new_k is not None:
        if "wq" in new_k or "wk" in new_k:
            new_state_dict[new_k] = unpermute(v)
        else:
            new_state_dict[new_k] = v


def save_model(n):
    os.makedirs(f"models/{n}B-alpaca", exist_ok=True)
    torch.save(new_state_dict, f"models/{n}B-alpaca/consolidated.00.pth")
    with open(f"models/{n}B-alpaca/params.json", "w") as f:
        json.dump(params, f)


if n in [7, 13, 30]:
    save_model(n)
