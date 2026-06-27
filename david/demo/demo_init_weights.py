"""A/B: what _init_weights does — with vs without post_init().

Run:
    python david/demo/demo_init_weights.py

  A  build layers, skip post_init()  →  _init_weights never runs
  B  build layers + post_init()      →  PreTrainedModel._init_weights runs
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_utils import PreTrainedModel


# ── Shared by A and B ─────────────────────────────────────────────────────────


class ToyConfig(PreTrainedConfig):
    model_type = "toy_demo"
    vocab_size: int = 32
    hidden_size: int = 16
    initializer_range: float = 0.02  # B reads this inside _init_weights


class ToyModel(PreTrainedModel):
    config_class = ToyConfig
    config: ToyConfig

    def __init__(self, config: ToyConfig, *, run_post_init: bool):
        super().__init__(config)
        # Shared: both A and B build the same layers
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # ── B only: this line triggers PreTrainedModel._init_weights ──────────
        # A passes run_post_init=False → this block is skipped
        if run_post_init:
            self.post_init()  # walks submodules → calls _init_weights(module)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.lm_head(self.embed(input_ids))


def _header(title: str) -> None:
    print(f"\n{'─' * 72}\n  {title}\n{'─' * 72}")


def _weight_stats(model: ToyModel) -> None:
    print(f"  embed.weight   std={model.embed.weight.std().item():.4f}")
    print(f"  lm_head.weight std={model.lm_head.weight.std().item():.4f}")


def main() -> None:
    print("A/B: _init_weights — the only difference is post_init()\n")
    config = ToyConfig(initializer_range=0.02)

    # ── A: build model WITHOUT post_init() ──────────────────────────────────
    # _init_weights is never called → PyTorch factory defaults stay in place
    torch.manual_seed(0)
    model_a = ToyModel(config, run_post_init=False)

    # ── B: build model WITH post_init() ─────────────────────────────────────
    # same seed, same layers — only difference is post_init() → _init_weights
    torch.manual_seed(0)
    model_b = ToyModel(config, run_post_init=True)

    # ── A: print weights (expect embed std ~ 1.0) ───────────────────────────
    _header("A  NO post_init() — _init_weights never called")
    print("  PyTorch factory defaults (Embedding ~ std 1.0)")
    _weight_stats(model_a)

    # ── B: print weights (expect embed std ~ initializer_range) ───────────────
    _header("B  WITH post_init() — PreTrainedModel._init_weights runs")
    print(f"  Uses config.initializer_range={config.initializer_range}")
    _weight_stats(model_b)

    print(f"\n{'─' * 72}")
    print("  Same class. Same seed. Only post_init() differs.")
    print("  post_init() → _init_weights(each module) → small controlled starting weights")
    print(f"{'─' * 72}\n")


if __name__ == "__main__":
    main()
