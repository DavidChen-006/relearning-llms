"""A/B: nn.Linear — with vs without _init_weights (via post_init).

Run:
    python david/demo/demo_linear_init_weights.py

  A  nn.Linear built, post_init SKIPPED  →  PyTorch factory weight/bias
  B  nn.Linear built, post_init CALLED   →  PreTrainedModel._init_weights fills them

Same layer shapes. Same forward input. Different starting matrices.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_utils import PreTrainedModel

torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)


class ToyConfig(PreTrainedConfig):
    model_type = "toy_demo"
    hidden_size: int = 4
    vocab_size: int = 6
    initializer_range: float = 0.02


class ToyModel(PreTrainedModel):
    config_class = ToyConfig
    config: ToyConfig

    def __init__(self, config: ToyConfig, *, run_post_init: bool):
        super().__init__(config)
        # Two nn.Linear layers — the module _init_weights cares about most
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        # ── B only: post_init() → _init_weights runs on each nn.Linear above ──
        if run_post_init:
            self.post_init()


def _header(title: str) -> None:
    print(f"\n{'─' * 72}\n  {title}\n{'─' * 72}")


def _show_linear(label: str, layer: nn.Linear) -> None:
    print(f"\n  {label}  (in={layer.in_features}, out={layer.out_features})")
    print(f"    weight.std:  {layer.weight.std().item():.4f}")
    print(f"    weight:\n{layer.weight.data}")
    if layer.bias is not None:
        print(f"    bias:\n{layer.bias.data}")
    else:
        print("    bias:         None")


def _forward_sample(model: ToyModel) -> torch.Tensor:
    x = torch.ones(1, 2, model.config.hidden_size)  # (batch, seq, in)
    return model.lm_head(model.proj(x))


def main() -> None:
    print("A/B: nn.Linear starting matrices — without vs with _init_weights\n")
    config = ToyConfig(hidden_size=4, vocab_size=6, initializer_range=0.02)
    print(f"  initializer_range (B uses this): {config.initializer_range}")

    # ── A: no post_init → _init_weights never runs on the Linear layers ───────
    torch.manual_seed(0)
    model_a = ToyModel(config, run_post_init=False)

    # ── B: post_init → _init_weights rewrites weight (and zeros bias) ─────────
    torch.manual_seed(0)
    model_b = ToyModel(config, run_post_init=True)

    _header("A  NO _init_weights — PyTorch factory defaults on nn.Linear")
    _show_linear("lm_head", model_a.lm_head)
    _show_linear("proj", model_a.proj)

    _header("B  WITH _init_weights — normal_(weight, std=0.02), zeros_(bias)")
    _show_linear("lm_head", model_b.lm_head)
    _show_linear("proj", model_b.proj)

    _header("Forward pass — same input, different outputs (because weights differ)")
    out_a = _forward_sample(model_a)
    out_b = _forward_sample(model_b)
    print(f"\n  input: ones(1, 2, {config.hidden_size})")
    print(f"  A output (lm_head ∘ proj):\n{out_a.data}")
    print(f"  B output (lm_head ∘ proj):\n{out_b.data}")
    print(f"\n  outputs equal? {torch.allclose(out_a, out_b)}")

    print(f"\n{'─' * 72}")
    print("  _init_weights input:  nn.Linear module (in place)")
    print("  _init_weights output: None — but weight/bias tensors are rewritten")
    print("  Then forward uses those tensors in:  output = input @ weight.T + bias")
    print(f"{'─' * 72}\n")


if __name__ == "__main__":
    main()
