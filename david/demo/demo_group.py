"""demo_group.py — grouping by HEAD vs grouping by TOKEN (same numbers, regrouped).

Tiny: 2 heads, 2 tokens, 2 dims. Watch transpose flip head-first <-> token-first.
"""
import torch

# attn_output right after attention: (heads, tokens, dim). Numbers 0..7 so you can track them.
attn = torch.arange(8).view(2, 2, 2)   # (2 heads, 2 tokens, 2 dim)

print("RAW tensor, shape (heads, tokens, dim) =", tuple(attn.shape))
print(attn)

print("\n" + "=" * 50)
print("GROUPED BY HEAD  (head-first) — how you RUN attention")
print("=" * 50)
for h in range(2):
    print(f"  HEAD {h}:")
    for t in range(2):
        print(f"      token{t} -> {attn[h, t].tolist()}")

# transpose heads<->tokens
attn_t = attn.transpose(0, 1)          # (2 tokens, 2 heads, 2 dim)

print("\n" + "=" * 50)
print("after .transpose -> shape (tokens, heads, dim) =", tuple(attn_t.shape))
print("GROUPED BY TOKEN (token-first) — how you MERGE")
print("=" * 50)
for t in range(2):
    print(f"  TOKEN {t}:")
    for h in range(2):
        print(f"      head{h} -> {attn_t[t, h].tolist()}")

# merge: glue each token's heads into one row
merged = attn_t.reshape(2, -1)         # (2 tokens, 4)
print("\n" + "=" * 50)
print("after .reshape(tokens, -1) — each token's heads glued together")
print("=" * 50)
for t in range(2):
    print(f"  token{t} -> {merged[t].tolist()}   (head0's 2 numbers + head1's 2 numbers)")

print("\nSame 8 numbers the whole time — only the GROUPING changed.")
