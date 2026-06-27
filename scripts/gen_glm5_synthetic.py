#!/usr/bin/env python3
"""Generate synthetic GLM-5.2 model weights for testing EP=8 training."""

import json
import os
import sys

import torch
from safetensors.torch import save_file

# ── Minimal GLM-5.2 config ──
H = 256               # hidden_size
N_LAYERS = 4
N_HEADS = 8
Q_LORA = 64
KV_LORA = 64
QK_NOPE = 32
QK_ROPE = 16
V_HEAD = 32
N_EXPERTS = 8         # divisible by 8 for EP=8
EXPERTS_PER_TOK = 2
N_SHARED = 1
MOE_INTER = 128
INTER = 128
FIRST_K_DENSE = 1     # layer 0 is dense
VOCAB = 1000
IDX_HEAD_DIM = 32
IDX_N_HEADS = 4
IDX_TOPK = 16
INDEXER_TYPES = ["full", "shared", "full", "shared"]

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/glm5-synthetic"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── config.json ──
config = {
    "model_type": "glm_moe_dsa",
    "architectures": ["GLM5ForCausalLM"],
    "hidden_size": H,
    "num_hidden_layers": N_LAYERS,
    "num_attention_heads": N_HEADS,
    "vocab_size": VOCAB,
    "tie_word_embeddings": False,
    "kv_lora_rank": KV_LORA,
    "q_lora_rank": Q_LORA,
    "qk_nope_head_dim": QK_NOPE,
    "qk_rope_head_dim": QK_ROPE,
    "v_head_dim": V_HEAD,
    "rope_theta": 8000000.0,
    "rms_norm_eps": 1e-6,
    "first_k_dense_replace": FIRST_K_DENSE,
    "n_routed_experts": N_EXPERTS,
    "num_experts_per_tok": EXPERTS_PER_TOK,
    "n_shared_experts": N_SHARED,
    "moe_intermediate_size": MOE_INTER,
    "intermediate_size": INTER,
    "scoring_func": "sigmoid",
    "n_group": 1,
    "topk_group": 1,
    "routed_scaling_factor": 1.0,
    "index_head_dim": IDX_HEAD_DIM,
    "index_n_heads": IDX_N_HEADS,
    "index_topk": IDX_TOPK,
    "indexer_types": INDEXER_TYPES,
    "index_topk_freq": 1,
    "index_skip_topk_offset": 0,
    "index_share_for_mtp_iteration": False,
    "rope_interleave": True,
    "expert_dtype": "bf16",
}

with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

# ── Generate weights ──
weights = {}

# Embed + head + final norm
weights["model.embed_tokens.weight"] = torch.randn(VOCAB, H, dtype=torch.bfloat16)
weights["model.norm.weight"] = torch.ones(H, dtype=torch.bfloat16)
weights["lm_head.weight"] = torch.randn(VOCAB, H, dtype=torch.bfloat16)

for layer in range(N_LAYERS):
    p = f"model.layers.{layer}"

    # Norms
    weights[f"{p}.input_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    weights[f"{p}.post_attention_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)

    # Attention
    weights[f"{p}.self_attn.wq_a.weight"] = torch.randn(Q_LORA, H, dtype=torch.bfloat16)
    weights[f"{p}.self_attn.wq_a_layernorm.weight"] = torch.ones(Q_LORA, dtype=torch.bfloat16)
    weights[f"{p}.self_attn.wq_b.weight"] = torch.randn(N_HEADS * (QK_NOPE + QK_ROPE), Q_LORA, dtype=torch.bfloat16)
    weights[f"{p}.self_attn.wkv.weight"] = torch.randn(KV_LORA + QK_ROPE, H, dtype=torch.bfloat16)
    weights[f"{p}.self_attn.wkv_a_layernorm.weight"] = torch.ones(KV_LORA, dtype=torch.bfloat16)
    weights[f"{p}.self_attn.wkv_b.weight"] = torch.randn(N_HEADS * (QK_NOPE + V_HEAD), KV_LORA, dtype=torch.bfloat16)
    weights[f"{p}.self_attn.wo.weight"] = torch.randn(H, N_HEADS * V_HEAD, dtype=torch.bfloat16)

    # Indexer (only for "full" layers)
    if INDEXER_TYPES[layer] == "full":
        idx_dim = IDX_N_HEADS * IDX_HEAD_DIM
        weights[f"{p}.self_attn.indexer.k_norm.weight"] = torch.ones(idx_dim, dtype=torch.bfloat16)
        weights[f"{p}.self_attn.indexer.k_norm.bias"] = torch.zeros(idx_dim, dtype=torch.bfloat16)
        weights[f"{p}.self_attn.indexer.weights_proj.weight"] = torch.randn(N_HEADS * IDX_HEAD_DIM, idx_dim, dtype=torch.bfloat16)
        weights[f"{p}.self_attn.indexer.wk.weight"] = torch.randn(idx_dim, KV_LORA, dtype=torch.bfloat16)
        weights[f"{p}.self_attn.indexer.wq_b.weight"] = torch.randn(N_HEADS * IDX_HEAD_DIM, Q_LORA, dtype=torch.bfloat16)

    # MLP
    if layer < FIRST_K_DENSE:
        # Dense layer
        weights[f"{p}.mlp.gate_proj.weight"] = torch.randn(INTER, H, dtype=torch.bfloat16)
        weights[f"{p}.mlp.up_proj.weight"] = torch.randn(INTER, H, dtype=torch.bfloat16)
        weights[f"{p}.mlp.down_proj.weight"] = torch.randn(H, INTER, dtype=torch.bfloat16)
    else:
        # MoE layer
        weights[f"{p}.mlp.gate.weight"] = torch.randn(N_EXPERTS, H, dtype=torch.bfloat16)
        weights[f"{p}.mlp.shared_experts.gate_proj.weight"] = torch.randn(MOE_INTER, H, dtype=torch.bfloat16)
        weights[f"{p}.mlp.shared_experts.up_proj.weight"] = torch.randn(MOE_INTER, H, dtype=torch.bfloat16)
        weights[f"{p}.mlp.shared_experts.down_proj.weight"] = torch.randn(H, MOE_INTER, dtype=torch.bfloat16)
        for e in range(N_EXPERTS):
            weights[f"{p}.mlp.experts.{e}.gate_proj.weight"] = torch.randn(MOE_INTER, H, dtype=torch.bfloat16)
            weights[f"{p}.mlp.experts.{e}.up_proj.weight"] = torch.randn(MOE_INTER, H, dtype=torch.bfloat16)
            weights[f"{p}.mlp.experts.{e}.down_proj.weight"] = torch.randn(H, MOE_INTER, dtype=torch.bfloat16)

# ── Save as safetensors ──
save_file(weights, os.path.join(OUTPUT_DIR, "model.safetensors"))

# ── tokenizer.json (minimal) ──
tokenizer = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [
        {"id": 0, "content": "<pad>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        {"id": 1, "content": "<s>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        {"id": 2, "content": "</s>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
    ],
    "normalizer": None,
    "pre_tokenizer": {"type": "Whitespace"},
    "post_processor": None,
    "decoder": None,
    "model": {
        "type": "WordLevel",
        "vocab": {f"tok{i}": i for i in range(VOCAB)},
        "unk_token": "<pad>"
    }
}

with open(os.path.join(OUTPUT_DIR, "tokenizer.json"), "w") as f:
    json.dump(tokenizer, f, indent=2)

print(f"Generated {len(weights)} tensors to {OUTPUT_DIR}")
print(f"Config: {N_LAYERS} layers, {N_EXPERTS} experts, H={H}, indexer_types={INDEXER_TYPES}")
