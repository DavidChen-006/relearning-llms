"""A/B/C: what nn.Module.__init__ actually does.

Run:
    python david/demo/demo_nn_module_init.py

  A  plain Python object
  B  after nn.Module.__init__ only  →  empty pegboard (no matrices yet)
  C  after adding nn.Linear         →  matrices appear + get registered
"""

from __future__ import annotations

import torch
import torch.nn as nn

torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)


def _header(title: str) -> None:
    print(f"\n{'─' * 72}\n  {title}\n{'─' * 72}")


def _show_module_internals(label: str, obj: nn.Module) -> None:
    print(f"\n  {label}")
    print(f"    type:              {type(obj).__name__}")
    print(f"    training:          {obj.training}")
    print(f"    child modules:     {dict(obj.named_children()) or '(none)'}")
    param_names = [name for name, _ in obj.named_parameters()]
    print(f"    parameters:        {param_names or '(none)'}")
    buffer_names = [name for name, _ in obj.named_buffers()]
    print(f"    buffers:           {buffer_names or '(none)'}")


class EmptyModule(nn.Module):
    """Only calls nn.Module.__init__ — no layers attached."""

    def __init__(self) -> None:
        super().__init__()  # ← this IS nn.Module.__init__


class ModuleWithLinear(nn.Module):
    """nn.Module.__init__ + one nn.Linear (matrices created here, not in super)."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 6, bias=True)  # ← weight matrix born here


def main() -> None:
    print("nn.Module.__init__ — empty container, not matrix factory\n")

    _header("A  Plain Python object (not a Module)")
    plain: object = object()
    print(f"  type: {type(plain).__name__}")
    print(f"  has .parameters()?  {hasattr(plain, 'parameters')}")
    print(f"  has .training?       {hasattr(plain, 'training')}")

    _header("B  nn.Module.__init__ only — EmptyModule()")
    empty = EmptyModule()
    _show_module_internals("right after super().__init__()", empty)
    print("\n  → Pegboard installed. No weight matrices exist yet.")

    _header("C  add nn.Linear after nn.Module.__init__ — ModuleWithLinear()")
    model = ModuleWithLinear()
    _show_module_internals("after self.linear = nn.Linear(4, 6)", model)
    print(f"\n  linear.weight.shape: {tuple(model.linear.weight.shape)}  (out × in)")
    print(f"  linear.weight:\n{model.linear.weight.data}")
    print(f"  linear.bias:   {model.linear.bias.data}")

    _header("What nn.Module.__init__ gives you (even when empty)")
    print("  .parameters()  → iterator over learnable tensors")
    print("  .named_modules() → walk the layer tree")
    print("  .to(device)     → move all tensors to GPU/CPU")
    print("  .state_dict()   → save/load weights")
    print("  .train() / .eval() → training vs inference mode")

    print(f"\n  empty.parameters():  {list(empty.parameters())}")
    print(f"  model.parameters():  {[tuple(p.shape) for p in model.parameters()]}")

    print(f"\n{'─' * 72}")
    print("  nn.Module.__init__  = empty toolbox with labeled slots")
    print("  nn.Linear(...)      = puts actual matrices in the slots")
    print(f"{'─' * 72}\n")


if __name__ == "__main__":
    main()
