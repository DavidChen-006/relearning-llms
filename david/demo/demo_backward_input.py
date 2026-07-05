"""demo_backward_input.py — INSPECT the input to loss.backward(): the recorded graph.

No backward() is called here. We only look at what `loss` secretly carries:
the grad_fn chain that forward left behind. That chain IS backward's input.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "architecture"))
from transformers import GlmMoeDsaConfig

from modeling_glm_moe_dsa import GlmMoeDsaForCausalLM

# tiny config so the graph is small enough to walk
config = GlmMoeDsaConfig(vocab_size=32, hidden_size=8, num_attention_heads=2,
                         intermediate_size=16, num_hidden_layers=1)
config._attn_implementation = "eager"
model = GlmMoeDsaForCausalLM(config)
torch.manual_seed(0)
ids = torch.randint(0, 32, (1, 4))

res = model(ids, labels=ids)            # forward + grade. NO backward anywhere in this file.
loss = res.loss

print("=" * 70)
print("1) what `loss` looks like from the outside")
print("=" * 70)
print("loss        :", loss)
print("loss.item() :", loss.item(), "  <- the number everyone talks about")
print("loss.grad_fn:", loss.grad_fn, "  <- the SECRET CARGO: tip of the recorded graph")
print("""
 -> `loss` is not a float. It is a tensor dragging the entire forward
    history behind it. That history is what backward() will consume.""")

print("=" * 70)
print("2) walk the chain: each node = one forward op, recorded in reverse")
print("=" * 70)
node = loss.grad_fn
for depth in range(12):
    if node is None:
        break
    print(f"   {'  ' * depth}{type(node).__name__}")
    # follow the first parent each time (the chain has branches; we walk one spine)
    parents = [fn for fn, _ in getattr(node, "next_functions", ()) if fn is not None]
    node = parents[0] if parents else None
print("   ... (keeps going all the way back to the embedding lookup)")
print("""
 -> read the names bottom-up as your own forward pass in reverse:
    NllLoss / LogSoftmax = cross_entropy, View = the flatten,
    Slice = the shift, then lm_head's matmul, the final norm, the layer...""")

print("=" * 70)
print("3) how BIG is the whole graph? (walk everything, count nodes)")
print("=" * 70)
seen = set()
stack = [loss.grad_fn]
leaf_params = 0
while stack:
    node = stack.pop()
    if node is None or node in seen:
        continue
    seen.add(node)
    if type(node).__name__ == "AccumulateGrad":     # a LEAF: where a parameter hangs
        leaf_params += 1
    stack.extend(fn for fn, _ in getattr(node, "next_functions", ()) if fn is not None)

n_params = sum(1 for _ in model.parameters())
print(f"graph nodes recorded by ONE forward pass : {len(seen)}")
print(f"AccumulateGrad leaves (blame mailboxes)  : {leaf_params}")
print(f"model.parameters() count                 : {n_params}   <- should match the mailboxes")
print("""
 -> every parameter has an AccumulateGrad node waiting at the graph's edge:
    that is the mailbox where backward() will deposit .grad. Right now,
    before any backward, every mailbox is empty:""")

w = model.model.layers[0].self_attn.q_proj.weight
print("   q_proj.weight.grad =", w.grad, "   <- None. Nothing delivered yet.")

print()
print("4) one node's saved homework: ops stash what they'll need for the chain rule")
mm = [n for n in seen if "Mm" in type(n).__name__]
print(f"   matmul nodes in the graph: {len(mm)} (each saved its input tensors for d(AB)=...)")
print("   e.g. the loss node saved:", [a for a in dir(loss.grad_fn) if a.startswith('_saved')])
