"""demo_shadow_matrix.py — how a weight MATRIX gets its shadow, and where shadows live.

PART 1: the recipe — W.grad = blame_at_output^T @ saved_input (one matmul, all verdicts)
PART 2: cell-level justice — every cell's blame = two numbers multiplied
PART 3: where shadows live — on each weight's own .grad slot, no central store
PART 4: who consumes them — the optimizer's parameter list; checkpoints never see grads
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "architecture"))
from transformers import GlmMoeDsaConfig

from modeling_glm_moe_dsa import GlmMoeDsaForCausalLM

torch.manual_seed(0)
torch.set_printoptions(precision=3, sci_mode=False)

# ============================================================== PART 1
print("=" * 68)
print("PART 1 — the recipe:  W.grad = blame_at_output^T @ saved_input")
print("=" * 68)
W = torch.randn(3, 3, requires_grad=True)      # a toy q_proj weight
x = torch.tensor([[1.0, 2.0, 3.0]])            # the input (this is what Mm SAVES)
q = x @ W.T                                    # the forward op under study
loss = (q ** 2).sum()
loss.backward()

blame_at_q = 2 * q.detach()                    # dloss/dq for this loss — the upstream blame
print("saved_input x       :", x.tolist()[0])
print("blame_at_output     :", [round(v, 3) for v in blame_at_q.tolist()[0]])
by_hand = blame_at_q.T @ x
print("\nby hand  (blame^T @ x):")
print(by_hand)
print("\nautograd (W.grad):")
print(W.grad)
print("\nmatch?", torch.allclose(by_hand, W.grad))
print(" -> same 3 ingredients as the scalar node: upstream blame x local slope")
print("    built from saved-self. At matrix scale the multiply IS a matmul.")

# ============================================================== PART 2
print("\n" + "=" * 68)
print("PART 2 — cell-level justice: every verdict = two numbers multiplied")
print("=" * 68)
print("W[i][j]'s forward job: multiply input j into output i. So its guilt:")
print("   W.grad[i][j] = blame_on_output_i  x  input_j\n")
for i, j in [(1, 2), (0, 0), (2, 1)]:
    v = blame_at_q[0, i].item() * x[0, j].item()
    print(f"   W.grad[{i}][{j}] = {blame_at_q[0, i].item():>8.3f} * {x[0, j].item():.1f} = {v:>8.3f}"
          f"   (autograd: {W.grad[i, j].item():>8.3f})")
print("\n   big input + badly-blamed output -> big verdict; row 2's whisper-blame")
print("   -> whisper verdicts. Nine micro-verdicts, one matmul.")

# ============================================================== PART 3
print("\n" + "=" * 68)
print("PART 3 — where shadows live: on their own weights. No central store.")
print("=" * 68)
config = GlmMoeDsaConfig(vocab_size=32, hidden_size=8, num_attention_heads=2,
                         intermediate_size=16, num_hidden_layers=1)
config._attn_implementation = "eager"
model = GlmMoeDsaForCausalLM(config)
ids = torch.randint(0, 32, (1, 4))
model(ids, labels=ids).loss.backward()          # one real backward on YOUR model

print(f"{'parameter':<44} {'weight shape':<14} {'.grad shape':<14} same?")
for name, p in model.named_parameters():
    print(f"{name:<44} {str(tuple(p.shape)):<14} {str(tuple(p.grad.shape)):<14} "
          f"{tuple(p.shape) == tuple(p.grad.shape)}")
print("\n -> 11 weights, 11 shadows, each hanging off its own tensor.")
print("    attention's verdicts live on attention; MLP's on MLP. Nothing filed anywhere.")

# ============================================================== PART 4
print("\n" + "=" * 68)
print("PART 4 — who consumes them, and what checkpoints actually contain")
print("=" * 68)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
n_registered = sum(len(g["params"]) for g in optimizer.param_groups)
print(f"optimizer was HANDED the parameter list at construction: {n_registered} tensors registered")

w = model.model.layers[0].self_attn.q_proj.weight
before = w.data.clone()
optimizer.step()                                # loops its list: each weight -= f(its OWN .grad)
print(f"after step():  q_proj.weight changed? {not torch.equal(before, w.data)}   (nudged in place)")

optimizer.zero_grad()
print(f"after zero_grad(): q_proj.weight.grad is {w.grad}   <- shadows are batch-scratch, wiped")

ckpt = model.state_dict()
print(f"\ncheckpoint = state_dict(): {len(ckpt)} entries, e.g. {list(ckpt)[0]!r}")
print("any '.grad' keys in it? ", any("grad" in k for k in ckpt))
print(" -> checkpoints save WEIGHTS for later. Gradients never touch disk:")
print("    they are born from a batch, consumed by step(), erased by zero_grad().")
