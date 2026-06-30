"""demo_mlp.py — watch ONE token flow through the (non-gated) MLP spine.

  out = down_proj( SiLU( up_proj(x) ) )
        up:   hidden(4) -> intermediate(8)   (expand)
        SiLU: fire / squash each hidden neuron
        down: intermediate(8) -> hidden(4)    (shrink back)
"""
import torch
from torch.nn.functional import silu

# ONE token: 4 features (whole numbers so it's readable)
x = torch.tensor([1., -2., 3., -1.])

# W_up : 4 -> 8  (each COLUMN is one hidden neuron's weights)
W_up = torch.tensor([
    [1., 0., -1., 2., 0., 1., -1., 0.],
    [0., 1., 1., -1., 2., 0., 0., 1.],
    [1., -1., 0., 1., 0., 2., 1., -1.],
    [-1., 0., 1., 0., 1., -1., 0., 2.],
])
# W_down : 8 -> 4
W_down = torch.tensor([
    [1., 0., 1., 0.],
    [0., 1., 0., 1.],
    [1., 1., 0., 0.],
    [0., 0., 1., 1.],
    [1., 0., 0., 1.],
    [0., 1., 1., 0.],
    [1., 0., 1., 0.],
    [0., 1., 0., 1.],
])

print("INPUT token x (4 features):", x.tolist())

h = x @ W_up                                   # up_proj
print("\n1) after up_proj (4 -> 8, EXPANDED):")
print("   ", [round(v, 2) for v in h.tolist()])

a = silu(h)                                    # activation
print("\n2) after SiLU (each neuron fires or gets squashed):")
for i in range(8):
    state = "FIRED " if a[i] > 0.5 else "  off "
    print(f"   neuron{i}: pre={h[i].item():5.1f}  ->  SiLU={a[i].item():6.2f}   [{state}]")

out = a @ W_down                               # down_proj
print("\n3) after down_proj (8 -> 4, back to token size):")
print("   ", [round(v, 2) for v in out.tolist()])

print("\nSo: 1 token in (4 nums) -> expand to 8 neurons -> some FIRE, some die -> shrink back to 4.")
print("Same shape out as in. The fired neurons decided what got written into the output.")
