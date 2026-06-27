"""nn.Linear in isolation — one module type, up close.

Run:
    python david/demo/demo_module.py

  nn.Linear = matrix multiply:  out = input @ weight.T + bias
  PreTrainedModel._init_weights touches Linear more than any other brick in big models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_utils import PreTrainedModel


class ToyConfig(PreTrainedConfig):
    model_type = "toy_demo"
    hidden_size: int = 16
    vocab_size: int = 32
    initializer_range: float = 0.02


class ToyModel(PreTrainedModel):
    """Only nn.Linear layers — no Embedding, no other types."""

    config_class = ToyConfig
    config: ToyConfig

    def __init__(self, config: ToyConfig):
        super().__init__(config)
        # in_features → out_features  (rows of the weight matrix)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            print(f"  _init_weights → nn.Linear  in={module.in_features}  out={module.out_features}")
        super()._init_weights(module)


def _header(title: str) -> None:
    print(f"\n{'─' * 72}\n  {title}\n{'─' * 72}")


def _describe_linear(name: str, layer: nn.Linear) -> None:
    print(f"  {name}")
    print(f"    type:         nn.Linear")
    print(f"    in_features:  {layer.in_features}")
    print(f"    out_features: {layer.out_features}")
    print(f"    weight.shape: {tuple(layer.weight.shape)}   # (out, in)")
    print(f"    bias:         {None if layer.bias is None else tuple(layer.bias.shape)}")
    print(f"    weight.std:   {layer.weight.std().item():.4f}")


def main() -> None:
    print("nn.Linear only — the module _init_weights hits most in real LLMs\n")

    _header("1. What is nn.Linear?")
    print("  A learned matrix:  output = input @ weight.T + bias")
    print("  weight shape is always (out_features, in_features)")

    _header("2. Build model (Linear layers only) — post_init() → _init_weights")
    model = ToyModel(ToyConfig())

    _header("3. Inspect each nn.Linear")
    _describe_linear("model.lm_head", model.lm_head)
    print()
    _describe_linear("model.proj", model.proj)

    _header("4. One forward step through a Linear")
    x = torch.randn(1, 3, model.config.hidden_size)  # (batch, seq, hidden)
    logits = model.lm_head(x)
    print(f"  input shape:  {tuple(x.shape)}")
    print(f"  output shape: {tuple(logits.shape)}  # last dim = out_features ({model.config.vocab_size})")

    _header("5. named_modules() — only root + Linear bricks")
    for name, mod in model.named_modules():
        label = name if name else "(root)"
        print(f"  {label:12s} → {type(mod).__name__}")

    print(f"\n{'─' * 72}")
    print("  In GLM: Q/K/V, MLP gates, lm_head, router — all nn.Linear.")
    print("  _init_weights: normal_(weight, std=initializer_range), zeros_(bias)")
    print(f"{'─' * 72}\n")


if __name__ == "__main__":
    main()
