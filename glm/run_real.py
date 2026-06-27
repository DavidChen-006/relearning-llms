"""Rung 0 (reverse learning) — stand up the REAL GLM-5.2 architecture at toy scale.

This is the real GlmMoeDsa model, just shrunk so it runs in seconds on a laptop.
Random weights => the output is gibberish; the point is that the whole thing RUNS.
"""
import torch
from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

# 1. Toy config: the real architecture, tiny numbers.
config = GlmMoeDsaConfig(
    vocab_size=96,
    hidden_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=4,
    n_routed_experts=8,
    num_experts_per_tok=2,
    n_shared_experts=1,
    first_k_dense_replace=1,
    intermediate_size=256,
    moe_intermediate_size=128,
    index_topk=64,
    max_position_embeddings=256,
)

# 2. Build the model from that shape and count parameters.
model = GlmMoeDsaForCausalLM(config)
n_params = sum(p.numel() for p in model.parameters())
print(f"model built — {n_params / 1e6:.2f}M parameters")

# 3. Fake input: a batch of random token ids, shape (B, T).
B, T = 2, 16
input_ids = torch.randint(0, config.vocab_size, (B, T))

# 4. Forward pass. Passing labels makes the model also return the loss.
out = model(input_ids=input_ids, labels=input_ids)
print("loss:", out.loss.item())
print("logits shape:", tuple(out.logits.shape))  # expect (B, T, vocab_size)

# 5. Sample 20 tokens from a 1-token prompt (gibberish — random weights).
gen = model.generate(input_ids[:, :1], max_new_tokens=20, do_sample=True)
print("generated ids:", gen[0].tolist())
