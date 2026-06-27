"""nn.Linear step-by-step — see every matrix and how data flows.

Run:
    python david/demo/demo_linear_flow.py

Traces:  output = input @ weight.T + bias
Prints each tensor at each step (small shapes so you can read them in the terminal).
"""

from __future__ import annotations

import torch
import torch.nn as nn

torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)


def _header(title: str) -> None:
    print(f"\n{'─' * 72}\n  {title}\n{'─' * 72}")


def _show(name: str, t: torch.Tensor) -> None:
    print(f"\n  {name}")
    print(f"    shape: {tuple(t.shape)}")
    print(f"    values:\n{t}")


def trace_linear_2d() -> None:
    """One row of data: input shape (in_features,) → output (out_features,)"""
    _header("Part 1 — single vector  input shape (in_features,)")

    in_features, out_features = 3, 2
    layer = nn.Linear(in_features, out_features, bias=True)

    # Fixed small weights so output is predictable
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 0.0, -1.0], [0.5, 0.5, 0.5]]))  # (out, in)
        layer.bias.copy_(torch.tensor([0.1, -0.2]))

    x = torch.tensor([2.0, 1.0, 3.0])  # (in_features,)

    _show("STEP 0 — input x", x)
    _show("STEP 1 — weight  (stored as out × in)", layer.weight)
    _show("STEP 2 — weight.T  (in × out)  ← we multiply by this", layer.weight.T)
    _show("STEP 3 — bias", layer.bias)

    matmul = x @ layer.weight.T
    _show("STEP 4 — matmul = x @ weight.T", matmul)

    result = matmul + layer.bias
    _show("STEP 5 — output = matmul + bias", result)

    pytorch = layer(x)
    _show("CHECK — layer(x) must match STEP 5", pytorch)
    print(f"\n  match: {torch.allclose(result, pytorch)}")


def trace_linear_3d() -> None:
    """Batch + sequence: (batch, seq, in) → (batch, seq, out) — same op on last dim"""
    _header("Part 2 — batch + sequence  input shape (batch, seq, in_features)")

    in_features, out_features = 3, 2
    layer = nn.Linear(in_features, out_features, bias=True)

    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 0.0, -1.0], [0.5, 0.5, 0.5]]))
        layer.bias.copy_(torch.tensor([0.1, -0.2]))

    # batch=2, seq=2, in=3
    x = torch.tensor(
        [
            [[2.0, 1.0, 3.0], [0.0, 1.0, 1.0]],   # sequence item 0 and 1 in batch 0
            [[1.0, 1.0, 1.0], [3.0, 0.0, 2.0]],   # batch 1
        ]
    )

    _show("STEP 0 — input x", x)
    print("\n  nn.Linear applies the SAME (in → out) map to the LAST dimension only.")
    print("  Each row of length `in_features` is one independent dot-product into `out_features`.")

    _show("STEP 1 — weight  (out × in)", layer.weight)
    _show("STEP 2 — weight.T", layer.weight.T)

    # PyTorch does: x @ weight.T + bias  (broadcast bias on last dim)
    matmul = x @ layer.weight.T
    _show("STEP 3 — matmul = x @ weight.T   shape (batch, seq, out)", matmul)

    result = matmul + layer.bias
    _show("STEP 4 — output = matmul + bias  (bias broadcasts)", result)

    pytorch = layer(x)
    _show("CHECK — layer(x)", pytorch)
    print(f"\n  match: {torch.allclose(result, pytorch)}")

    _header("Part 2b — zoom one position: x[0, 0, :] → output[0, 0, :]")
    row = x[0, 0]
    _show("  input row x[0,0,:]", row)
    row_out = row @ layer.weight.T + layer.bias
    _show("  manual row @ weight.T + bias", row_out)
    _show("  output[0,0,:] from layer", pytorch[0, 0])
    print(f"\n  match: {torch.allclose(row_out, pytorch[0, 0])}")


def trace_no_bias() -> None:
    _header("Part 3 — no bias  (like lm_head in GLM)")

    layer = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 0.0, -1.0], [0.5, 0.5, 0.5]]))

    x = torch.tensor([2.0, 1.0, 3.0])
    _show("input x", x)
    _show("weight", layer.weight)
    _show("output = x @ weight.T  (no bias term)", x @ layer.weight.T)
    _show("layer(x)", layer(x))


def main() -> None:
    print("nn.Linear data flow — every matrix printed\n")
    print("  Formula:  output = input @ weight.T + bias")
    print("  weight stored as (out_features, in_features)")

    trace_linear_2d()
    trace_linear_3d()
    trace_no_bias()

    print(f"\n{'─' * 72}")
    print("  Flow:  input  →  @ weight.T  →  + bias  →  output")
    print("  weight.T is why shape is (out, in) in memory but (in) dots into (out).")
    print(f"{'─' * 72}\n")


if __name__ == "__main__":
    main()
