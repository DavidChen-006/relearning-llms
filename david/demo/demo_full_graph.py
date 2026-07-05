"""demo_full_graph.py — the WHOLE iceberg: every node of the backward graph, annotated.

The earlier demos walked one spine (9 nodes). This prints the full tree:
  - every node, indented by depth (children = the ops that ran BEFORE it in forward)
  - each node annotated with the forward operation it is the twin of
  - every AccumulateGrad leaf labeled with the PARAMETER it is the mailbox for —
    the parameter names (model.layers.0.self_attn.q_proj.weight ...) ARE the block map
  - repeated subtrees (residual joins) printed once, then referenced

Config is 1 layer so the whole thing fits in a terminal.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "architecture"))
from transformers import GlmMoeDsaConfig

from modeling_glm_moe_dsa import GlmMoeDsaForCausalLM

config = GlmMoeDsaConfig(vocab_size=32, hidden_size=8, num_attention_heads=2,
                         intermediate_size=16, num_hidden_layers=1)
config._attn_implementation = "eager"
model = GlmMoeDsaForCausalLM(config)
torch.manual_seed(0)
ids = torch.randint(0, 32, (1, 4))
loss = model(ids, labels=ids).loss

# ---- what each node type means: the forward op it is the backward twin of ----
MEANING = {
    "NllLossBackward0": "cross_entropy 2/2: pick correct, -log, mean",
    "LogSoftmaxBackward0": "cross_entropy 1/2: scores -> log-probs",
    "ViewBackward0": "reshape",
    "UnsafeViewBackward0": "reshape (internal)",
    "SliceBackward0": "the [:, :-1] SHIFT / slicing",
    "MmBackward0": "matmul (2D)",
    "BmmBackward0": "matmul (batched) - the QK^T or weights@V",
    "AddBackward0": "add (residual or +mask)",
    "MulBackward0": "multiply (norm gamma / RoPE / scaling)",
    "SoftmaxBackward0": "softmax (attention weights)",
    "PowBackward0": "x^2 (RMSNorm variance)",
    "MeanBackward1": "mean (RMSNorm variance)",
    "RsqrtBackward0": "1/sqrt (RMSNorm)",
    "SiluBackward0": "SiLU activation (MLP)",
    "TransposeBackward0": "transpose (head split/merge)",
    "ExpandBackward0": "broadcast",
    "CloneBackward0": "contiguous copy",
    "ToCopyBackward0": "dtype/device cast",
    "EmbeddingBackward0": "embed_tokens: id -> vector lookup",
    "CatBackward0": "concat (rotate_half)",
    "NegBackward0": "negate (rotate_half's -x2)",
    "AccumulateGrad": "MAILBOX",
    "TBackward0": "transpose (linear weight)",
    "AddmmBackward0": "matmul+bias (linear)",
    "SqueezeBackward1": "squeeze",
    "UnsqueezeBackward0": "unsqueeze",
}

# map each parameter tensor -> its dotted name, so mailboxes identify themselves
param_names = {p: n for n, p in model.named_parameters()}

seen = {}
counter = {"n": 0}
mailboxes = []


def walk(node, depth):
    if node is None:
        return
    pad = "· " * min(depth, 30)
    name = type(node).__name__
    if node in seen:
        print(f"{pad}({name}  -> already shown as #{seen[node]}: shared path, e.g. a residual join)")
        return
    counter["n"] += 1
    idx = counter["n"]
    seen[node] = idx
    if name == "AccumulateGrad":
        pname = param_names.get(node.variable, "?")
        mailboxes.append(pname)
        print(f"{pad}#{idx:<3} [MAILBOX] {pname}")
        return
    print(f"{pad}#{idx:<3} {name:<24} = {MEANING.get(name, '?')}")
    for parent, _slot in node.next_functions:
        walk(parent, depth + 1)


print("FULL BACKWARD GRAPH  (top = loss; deeper = earlier in forward; leaves = weights)")
print("=" * 90)
walk(loss.grad_fn, 0)

print("=" * 90)
print(f"total nodes: {counter['n']}   mailboxes: {len(mailboxes)}")
print("\nTHE BLOCK MAP — every mailbox, grouped by the block its name reveals:")
blocks = {}
for p in mailboxes:
    if "self_attn" in p:
        key = "ATTENTION      (model.layers.0.self_attn.*)"
    elif "mlp" in p:
        key = "MLP            (model.layers.0.mlp.*)"
    elif "layernorm" in p:
        key = "LAYER NORMS    (model.layers.0.*layernorm)"
    elif "embed" in p:
        key = "EMBEDDINGS     (model.embed_tokens)"
    elif "lm_head" in p:
        key = "LM HEAD"
    else:
        key = "FINAL NORM     (model.norm)"
    blocks.setdefault(key, []).append(p.split(".")[-2] + "." + p.split(".")[-1])
for k in sorted(blocks):
    print(f"   {k:<45} {blocks[k]}")
print("""
HOW TO READ THE TREE:
  depth 0-4    = the grading tail (cross-entropy, shift)         <- the old 9-node spine
  the LM HEAD fork = first driveway (its weight mailbox)
  below that   = final norm, then the DECODER LAYER opens up:
                 residual adds -> MLP ops (Silu, matmuls) -> norm ops (Pow/Mean/Rsqrt)
                 -> ATTENTION ops (Bmm=QK^T & weights@V, Softmax, Transpose=heads,
                    Cat/Neg/Mul=RoPE's rotate_half) -> input norm -> EMBEDDING lookup.
  '(already shown)' lines are where two roads merge - the residual stream's joins.""")
