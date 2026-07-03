"""demo_rope_identity.py — proof that rotating does NOT destroy a vector's identity.

The claim being tested: "moving a point in space completely changes its identity."
The counter-claim: identity is RELATIONAL (lengths + dot products against other vectors),
rotation preserves relations exactly, and the only relational change RoPE introduces is
the deliberate one — a twist of exactly (m - n).

All in 2D so every number is readable; the 128-dim case is 64 of these planes at once.
"""
import math

import torch

torch.set_printoptions(precision=4, sci_mode=False)


def rot(v, deg):
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return torch.tensor([v[0] * c - v[1] * s, v[1] * c + v[0] * s])


# ---------- PROOF 1: the constellation — rotate EVERYTHING, nothing visible changes ----------
print("=" * 70)
print("PROOF 1: rotate a whole constellation of random vectors by the SAME angle")
print("=" * 70)
torch.manual_seed(0)
stars = [torch.randn(2) for _ in range(4)]                 # 4 random 'super distributed' points
turned = [rot(v, 77.0) for v in stars]                     # rigid turn, arbitrary angle

print("\nall pairwise dot products (the only thing attention can ever see):")
print(f"{'pair':>8} {'before':>10} {'after':>10}")
for i in range(4):
    for j in range(i, 4):
        b = torch.dot(stars[i], stars[j]).item()
        a = torch.dot(turned[i], turned[j]).item()
        print(f"  v{i}.v{j}   {b:10.4f} {a:10.4f}")
print(" -> IDENTICAL columns. The points all moved, the constellation didn't.")
print("    Identity = relationships, and a rigid turn preserves every one of them.")

# ---------- PROOF 2: the OLD way (adding a position vector) really does dilute ----------
print("\n" + "=" * 70)
print("PROOF 2: the additive approach (old models) DOES corrupt the comparison")
print("=" * 70)
q, k = stars[0], stars[1]
pos_vec = torch.tensor([0.9, -0.4])                        # some 'position 5' vector to ADD
q_add, k_add = q + pos_vec, k + pos_vec
print(f"\n  q.k before adding position vector: {torch.dot(q, k).item():9.4f}")
print(f"  q.k after  adding position vector: {torch.dot(q_add, k_add).item():9.4f}   <- changed!")
print(f"  |q| before: {q.norm().item():.4f}   after: {q_add.norm().item():.4f}   <- length changed too")
print(" -> content similarity and lengths both distorted: position DILUTES identity.")

# ---------- PROOF 3: RoPE = same rotation cancels, different rotations leave ONLY the gap ----------
print("\n" + "=" * 70)
print("PROOF 3: RoPE's twist — the only change is the deliberate one: the gap (m - n)")
print("=" * 70)
SPEED = 10.0                                               # degrees per position (one 'dial')
print(f"\n  same content q, k placed at positions m and n (speed {SPEED} deg/position):")
print(f"{'m':>4} {'n':>4} {'gap':>5} {'rotated q.k':>13}")
for m, n in [(0, 0), (5, 5), (9, 9), (3, 1), (7, 5), (12, 10), (5, 0), (10, 5)]:
    qr, kr = rot(q, m * SPEED), rot(k, n * SPEED)
    print(f"{m:>4} {n:>4} {m - n:>5} {torch.dot(qr, kr).item():13.4f}")
print(f"  (unrotated q.k = {torch.dot(q, k).item():.4f})")
print(" -> equal positions (gap 0): dot product EXACTLY unchanged, wherever they sit.")
print(" -> gap 2 pairs all agree; gap 5 pairs all agree: score = content x twist(gap).")
print("    Absolute position cancelled; only the offset David asked for survives.")

# ---------- PROOF 4: invertible = nothing was destroyed ----------
print("\n" + "=" * 70)
print("PROOF 4: rotate back and recover the original, bit for bit")
print("=" * 70)
v = stars[2]
there = rot(v, 123.0)
back = rot(there, -123.0)
print(f"\n  original     : {v.tolist()}")
print(f"  rotated +123 : {[round(x, 4) for x in there.tolist()]}")
print(f"  rotated -123 : {[round(x, 4) for x in back.tolist()]}")
print(f"  recovered exactly? {torch.allclose(v, back, atol=1e-6)}")
print(" -> an undoable transformation cannot have destroyed information.")
