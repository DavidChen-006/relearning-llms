"""A/B: same tiny LM — plain nn.Module vs PreTrainedModel.

Run:
    python david/demo/demo_pretrained_model.py

Same layers (embed → norm → lm_head). Only the base class differs.
Side-by-side shows what PreTrainedModel actually adds.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.init as init
from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_utils import PreTrainedModel


class ToyConfig(PreTrainedConfig):
    model_type = "toy_demo"
    vocab_size: int = 32
    hidden_size: int = 16


# ── A: plain PyTorch — no HF glue ────────────────────────────────────────────


class PlainToyCausalLM(nn.Module):
    """Same architecture as B, but just nn.Module."""

    def __init__(self, config: ToyConfig):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # No post_init(). No _init_weights(). PyTorch default init only.

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.lm_head(self.norm(self.embed(input_ids)))


# ── B: HuggingFace PreTrainedModel — adds config + init + save/load ───────────


class ToyForCausalLM(PreTrainedModel):
    """Same architecture as A, plus HF machinery."""

    config_class = ToyConfig
    config: ToyConfig

    def __init__(self, config: ToyConfig):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()  # walks all submodules → calls _init_weights()

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.lm_head(self.norm(self.embed(input_ids)))

    def _init_weights(self, module: nn.Module) -> None:
        super()._init_weights(module)
        if isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=0.02)


# ── Demo helpers ──────────────────────────────────────────────────────────────


def _header(title: str) -> None:
    print(f"\n{'─' * 72}\n  {title}\n{'─' * 72}")


def _try(label: str, fn) -> None:
    print(f"\n  {label}")
    try:
        result = fn()
        print(f"    OK → {result!r}")
    except Exception as e:
        print(f"    {type(e).__name__}: {e}")


def _embed_std(model: nn.Module) -> float:
    return model.embed.weight.std().item()


def main() -> None:
    print("A/B: plain nn.Module  vs  PreTrainedModel")
    print("Same embed → norm → lm_head.  A = PlainToyCausalLM  |  B = ToyForCausalLM\n")

    config = ToyConfig(vocab_size=32, hidden_size=16)
    input_ids = torch.tensor([[1, 2, 3]])

    torch.manual_seed(0)
    model_a = PlainToyCausalLM(config)
    torch.manual_seed(0)
    model_b = ToyForCausalLM(config)

    _header("1. Weight init — B calls post_init() → _init_weights()")
    print(f"  A  embed.weight std: {_embed_std(model_a):.4f}  (PyTorch default)")
    print(f"  B  embed.weight std: {_embed_std(model_b):.4f}  (our _init_weights: std=0.02)")
    print("  → Same seed, different weights — B's custom init actually ran.")

    _header("2. Config binding — B keeps config on the model")
    _try("A  model.config", lambda: model_a.config)
    _try("B  model.config.hidden_size", lambda: model_b.config.hidden_size)

    _header("3. save_pretrained / from_pretrained — B only")
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "toy-demo"

        _try("A  model.save_pretrained(...)", lambda: model_a.save_pretrained(ckpt))

        def save_and_reload_b():
            model_b.save_pretrained(ckpt)
            files = sorted(p.name for p in ckpt.iterdir())
            reloaded = ToyForCausalLM.from_pretrained(ckpt)
            with torch.no_grad():
                diff = (model_b(input_ids) - reloaded(input_ids)).abs().max().item()
            return f"files={files}, max_logit_diff={diff:.2e}"

        _try("B  model.save_pretrained(...) → from_pretrained(...)", save_and_reload_b)

    _header("4. Forward pass — both compute logits (values differ from test 1)")
    with torch.no_grad():
        logits_a = model_a(input_ids)
        logits_b = model_b(input_ids)
    print(f"  A  logits shape: {tuple(logits_a.shape)}  mean: {logits_a.mean():.4f}")
    print(f"  B  logits shape: {tuple(logits_b.shape)}  mean: {logits_b.mean():.4f}")
    print("  → Architecture works either way. PreTrainedModel adds init + config + save/load.")

    print(f"\n{'─' * 72}")
    print("  Takeaway: PreTrainedModel does NOT replace nn.Module layers.")
    print("  It adds: (1) config on self  (2) _init_weights via post_init  (3) save/load API")
    print(f"{'─' * 72}\n")


if __name__ == "__main__":
    main()
