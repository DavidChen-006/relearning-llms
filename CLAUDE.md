# GLM-5.2 From Scratch — Reverse-Learning Build Plan

Understand GLM-5.2 by **starting from the real model and peeling it apart**, top-down, until
you hit the fundamentals at the bottom. Learn by deconstruction, not construction.

---

## The approach: reverse learning

Instead of building up from a bigram to the full model, we go the other way:

1. **Stand up the real `GlmMoeDsa` model** at toy scale — get the whole thing running as a
   black box you don't fully understand yet. That's fine.
2. **Peel one component at a time**, from the outer skeleton inward, down to the raw math.
3. **Prove understanding by swap-in:** at each rung you reimplement one component from scratch
   and drop it into the real model. **If the outputs still match, you understood it.**

> Top-down means you run code you don't fully grasp at first; understanding comes from zooming
> in. The swap-in gate is what keeps it rigorous — no hand-waving allowed.

---

## Goal & honest scope

| | |
|---|---|
| **In scope** | Understand and reimplement the full *architecture* by deconstructing the real one, at toy scale. Toy GLM-5.2 and real GLM-5.2 are the **same code** — only config numbers + data are toy. |
| **Out of scope** | Pretraining the real 744B weights (28.5T tokens, ~$millions). We don't reproduce the released weights. |
| **Capstone** | Load the **real released weights** into your fully-rebuilt version and confirm logits match the reference. Proof of correctness without training. |

---

## The reference (what we deconstruct)

- `../glm 5.2/transformers_glm_moe_dsa/modeling_glm_moe_dsa.py` — the forward pass (the target)
- `../glm 5.2/transformers_glm_moe_dsa/configuration_glm_moe_dsa.py` — the shape (already read)

Rule: attempt each swap-in **yourself first**, then compare against the reference.

---

## The Reverse Ladder

Peel from the roof down to the foundation. Each rung: understand a component, then reimplement
it and swap it into the real model — outputs must match.

| Rung | You take apart | Reimplement (the file you create) | Swap-in verify ✅ |
|---|---|---|---|
| **R0** | **Stand up the real model** — instantiate a tiny `GlmMoeDsa`, run a forward pass, get logits + loss, sample. Black box that *works*. | `run_real.py` (harness, not architecture) | runs end-to-end; loss is finite; it samples text |
| **R1** | **Outer skeleton** — `embed_tokens → [layers] → norm → lm_head` + the residual stream | `skeleton.py` | trace one token's shape through the whole pipe |
| **R2** | **One decoder layer** — attn sublayer + MLP sublayer + residuals + norms | `layer.py` | your layer wrapper's output matches the real layer |
| **R3** | **The MLP path** — dense SwiGLU, then **MoE** (router, experts, shared expert) | `moe.py` | your MoE output matches the real MoE |
| **R4** | **The attention path** — Q/K/V + softmax, then **MLA**, then RoPE | `mla.py` | your attention output matches the real one |
| **R5** | **DSA** — the sparse-attention indexer + IndexShare | `dsa.py` | your sparse attn ≈ real sparse attn on toy data |
| **R6** | **The primitives** — RMSNorm, RoPE math, matmul/softmax, cross-entropy | `primitives.py` | each raw-torch reimpl matches its reference op |
| **★ Finish** | **Verify against reality** — load real released weights into your fully-rebuilt model | `verify.py` | your logits match the reference within tolerance |

---

## "Entirely finished" =

1. Every component of `GlmMoeDsa` has a from-scratch reimplementation **you wrote**, and
2. each one **passed its swap-in test** (outputs match the real model), and
3. `verify.py` shows the fully-rebuilt model **matches the reference on real weights**.

At that point you've taken GLM-5.2 apart to the bolts and put it back together — and proven it.

---

## Toy vs real scale (keep it honest)

Same code, different numbers. Start tiny so each rung runs in seconds on the laptop (mps).

| Field | Toy (you) | Real GLM-5.2 |
|---|---|---|
| `hidden_size` | 128 | 6144 |
| `num_hidden_layers` | 4 | 78 |
| `num_attention_heads` | 4 | 64 |
| `n_routed_experts` / `num_experts_per_tok` | 8 / 2 | 256 / 8 |
| `index_topk` | 64 | 2048 |
| `max_position_embeddings` | 256 | 202,752 |
| `vocab_size` | ~96 (chars) | 154,880 |

When every component passes swap-in at toy scale, the only thing between you and the real model
is compute — no longer a learning problem.
