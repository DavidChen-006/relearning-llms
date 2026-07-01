"""demo_causal_mask.py — proof the mask works: staircase -inf -> softmax -> weights @ V.

Recipe view: each ROW of the weights is a recipe; each ROW of V is an ingredient (a value
vector). out[i] = weighted sum of V's rows using recipe i. The mask zeros the FUTURE
ingredients, so a token's output is built only from itself + the past.
"""
import torch
from torch.nn.functional import softmax

S = 4
torch.manual_seed(0)

scores = torch.randn(S, S)                                  # raw QK^T/sqrt(d) scores
mask = torch.full((S, S), float("-inf")).triu(1)           # -inf ABOVE the diagonal (the future)
weights = softmax(scores + mask, dim=-1)                    # -inf -> 0 after softmax

# V: token j's value vector = all j*10, so you can SEE which ingredients made it into each output
V = torch.tensor([[10., 10, 10], [20, 20, 20], [30, 30, 30], [40, 40, 40]])
out = weights @ V

torch.set_printoptions(precision=2, sci_mode=False)
print("MASK (added to scores) — -inf above the diagonal:\n", mask)
print("\nWEIGHTS = softmax(scores + mask) — note UPPER TRIANGLE IS 0, rows sum to 1:")
for i in range(S):
    print(f"  token{i}: {[round(w, 2) for w in weights[i].tolist()]}  sum={weights[i].sum():.2f}")

print("\nV (ingredients, one value vector per token):\n", V)
print("\nout = weights @ V:\n", out)

print("\n--- RECIPE breakdown (out[i] = sum of weight * V-row) ---")
for i in range(S):
    print(f"\n  token{i} recipe = {[round(w, 2) for w in weights[i].tolist()]}")
    for j in range(S):
        tag = "FUTURE -> 0" if j > i else "used"
        print(f"     {weights[i, j]:.2f} * V[{j}]{V[j].tolist()} = {[round(v, 1) for v in (weights[i, j] * V[j]).tolist()]}   ({tag})")
    print(f"     = out[{i}] = {[round(v, 1) for v in out[i].tolist()]}")

print("\n--- PROOF IN THE PUDDING: change a FUTURE ingredient, earlier outputs stay identical ---")
V2 = V.clone()
V2[3] = torch.tensor([999., 999, 999])     # blow up token 3's value (a future ingredient)
out2 = weights @ V2
print("out  (V[3]=40s) token0..2:", [[round(v, 1) for v in out[i].tolist()] for i in range(3)])
print("out2 (V[3]=999) token0..2:", [[round(v, 1) for v in out2[i].tolist()] for i in range(3)])
print("token 0,1,2 outputs identical?", torch.allclose(out[:3], out2[:3]),
      "  <- the future ingredient had 0 weight, so it changed nothing")
