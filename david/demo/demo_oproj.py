"""demo_oproj.py — what does o_proj's "blending" actually do?

Glued heads are just chunks sitting next to each other. o_proj (a matrix multiply)
makes every OUTPUT number a mix of ALL the heads' numbers. That mixing = "blending".
"""
import torch

# Glued head output for ONE token: head0 = [1, 2], head1 = [10, 20]
glued = torch.tensor([1.0, 2.0, 10.0, 20.0])     # [head0 | head1]
print("glued (before o_proj):", glued.tolist())
print("   first 2 = head0 only:", glued[:2].tolist())
print("   last  2 = head1 only:", glued[2:].tolist())
print("   -> right now each number belongs to ONE head. No mixing yet.\n")

# o_proj is a learned weight matrix (4x4 here). Each column makes one output number.
Wo = torch.tensor([
    [1.0, 0.0, 0.5, 1.0],
    [0.0, 1.0, 0.5, 0.0],
    [1.0, 0.0, 0.5, 0.0],
    [0.0, 1.0, 0.5, 1.0],
])

out = glued @ Wo
print("o_proj output:", out.tolist())

print("\nWhy each output number is a BLEND of BOTH heads:")
for j in range(4):
    parts = [f"{glued[i].item():.0f}*{Wo[i, j].item():.1f}" for i in range(4)]
    print(f"  out[{j}] = {' + '.join(parts)} = {out[j].item():.1f}")
    print("          (uses head0's 1,2 AND head1's 10,20 -> mixed)")

print("\nBefore: out[0] would just be head0's stuff.")
print("After o_proj: every out number mixes head0 AND head1 together. THAT is blending.")
