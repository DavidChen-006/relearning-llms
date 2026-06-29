"""demo_multihead.py — single-head vs multi-head attention, side by side.

Same 4 tokens, same features. The point you'll SEE:
  single head  -> ONE attention pattern
  multi-head   -> SEVERAL DIFFERENT attention patterns (more viewpoints, same tokens)
"""
import torch
from torch.nn.functional import softmax

torch.manual_seed(0)

S, D = 4, 8          # 4 tokens, 8 features each
H = 4                # 4 heads -> each head gets D/H = 2 features
d_head = D // H

x = torch.randn(S, D)                       # the 4 input tokens
Wq = torch.randn(D, D)
Wk = torch.randn(D, D)

Q = x @ Wq                                  # (4, 8)
K = x @ Wk                                  # (4, 8)


def show(name, pattern):
    print(f"\n{name}  (rows=query token, cols=key token, each row sums to 1):")
    for i, row in enumerate(pattern.tolist()):
        print(f"   token{i}: {[round(v, 2) for v in row]}")


# ===================== SINGLE-HEAD =====================
print("=" * 55)
print("SINGLE-HEAD: one Q/K using all 8 features -> ONE pattern")
print("=" * 55)
scores = (Q @ K.T) / (D ** 0.5)             # (4,4) using full 8-dim vectors
show("the ONE attention pattern", softmax(scores, dim=-1))

# ===================== MULTI-HEAD =====================
print("\n" + "=" * 55)
print(f"MULTI-HEAD: split 8 features into {H} heads of {d_head} -> {H} patterns")
print("=" * 55)
Qh = Q.view(S, H, d_head)                   # (4 tokens, 4 heads, 2 feats)
Kh = K.view(S, H, d_head)
for h in range(H):
    qh = Qh[:, h, :]                        # this head's Q  (4, 2)
    kh = Kh[:, h, :]                        # this head's K  (4, 2)
    scores_h = (qh @ kh.T) / (d_head ** 0.5)
    show(f"head {h} pattern", softmax(scores_h, dim=-1))

print("\n" + "=" * 55)
print("Notice: the single-head gives 1 pattern; the 4 heads give 4 DIFFERENT")
print("patterns over the SAME 4 tokens. Same tokens, more viewpoints.")
print("=" * 55)
