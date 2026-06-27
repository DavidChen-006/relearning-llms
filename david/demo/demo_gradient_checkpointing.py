"""A/B: GradientCheckpointingLayer — train with less GPU memory.

Run:
    python david/demo/demo_gradient_checkpointing.py

  GradientCheckpointingLayer = nn.Module whose __call__ can re-run forward on backward
  instead of storing every activation (trades compute for memory).

  GLM:  class GlmMoeDsaDecoderLayer(GradientCheckpointingLayer)
        model.gradient_checkpointing_enable()  # training only
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import GradientCheckpointingLayer
from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_utils import PreTrainedModel


class ToyConfig(PreTrainedConfig):
    model_type = "toy_demo"
    hidden_size: int = 256
    num_hidden_layers: int = 8
    vocab_size: int = 32


# ── A: plain block — always stores activations during forward ─────────────────


class PlainBlock(nn.Module):
    """Same math as ToyBlock, but no checkpoint hook."""

    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ── B: GradientCheckpointingLayer — can checkpoint when flag is on ────────────


class ToyBlock(GradientCheckpointingLayer):
    """Mimics GlmMoeDsaDecoderLayer: inherit GradientCheckpointingLayer, not nn.Module."""

    gradient_checkpointing = False  # class default; HF flips to True via enable()

    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ToyModel(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = ToyConfig
    config: ToyConfig

    def __init__(self, config: ToyConfig, *, block_cls: type[nn.Module]):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [block_cls(config.hidden_size) for _ in range(config.num_hidden_layers)]
        )
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            # positional x — required when checkpointing (see GradientCheckpointingLayer doc)
            x = layer(x)
        return self.head(x)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed


def _header(title: str) -> None:
    print(f"\n{'─' * 72}\n  {title}\n{'─' * 72}")


def _peak_mem_mb() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1e6
    # CPU: no allocator stats — return 0 as placeholder
    return 0.0


def _train_step(model: PreTrainedModel, input_ids: torch.LongTensor) -> float:
    model.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    logits = model(input_ids)
    loss = logits.pow(2).mean()
    loss.backward()
    return _peak_mem_mb()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GradientCheckpointingLayer demo  (device={device})\n")

    config = ToyConfig(hidden_size=256, num_hidden_layers=8)
    B, T = 4, 128
    input_ids = torch.randint(0, config.vocab_size, (B, T), device=device)

    _header("1. What GradientCheckpointingLayer adds")
    block = ToyBlock(config.hidden_size)
    print(f"  isinstance GradientCheckpointingLayer: {isinstance(block, GradientCheckpointingLayer)}")
    print(f"  block.gradient_checkpointing (default): {block.gradient_checkpointing}")
    print("  When True + model.training: __call__ wraps forward in torch.utils.checkpoint")
    print("  → backward re-runs forward instead of keeping all activations in memory")

    _header("2. ToyBlock is a drop-in base class — same forward when checkpoint OFF")
    toy_model = ToyModel(config, block_cls=ToyBlock).to(device)
    toy_model.eval()
    with torch.no_grad():
        out = toy_model(input_ids)
    print(f"  logits shape: {tuple(out.shape)}")
    print("  ToyBlock.forward unchanged — GradientCheckpointingLayer only hooks __call__")

    _header("3. A  checkpoint OFF  (PlainBlock stack)")
    plain_model = ToyModel(config, block_cls=PlainBlock).to(device)
    mem_a = _train_step(plain_model, input_ids)
    if device == "cuda":
        print(f"  peak GPU memory during backward: {mem_a:.1f} MB")
    else:
        print("  (no CUDA — run on GPU to see memory difference)")

    _header("4. B  checkpoint OFF  (ToyBlock, flag still False)")
    toy_off = ToyModel(config, block_cls=ToyBlock).to(device)
    mem_b_off = _train_step(toy_off, input_ids)
    if device == "cuda":
        print(f"  peak GPU memory during backward: {mem_b_off:.1f} MB")

    _header("5. B  checkpoint ON   model.gradient_checkpointing_enable()")
    toy_on = ToyModel(config, block_cls=ToyBlock).to(device)
    toy_on.gradient_checkpointing_enable()
    print(f"  layer[0].gradient_checkpointing: {toy_on.layers[0].gradient_checkpointing}")
    print(f"  model.is_gradient_checkpointing: {toy_on.is_gradient_checkpointing}")
    mem_b_on = _train_step(toy_on, input_ids)
    if device == "cuda":
        print(f"  peak GPU memory during backward: {mem_b_on:.1f} MB")
        print(f"  saved ≈ {mem_a - mem_b_on:.1f} MB vs plain (A)")

    _header("6. How GLM wires it (your modeling file)")
    print("  class GlmMoeDsaDecoderLayer(GradientCheckpointingLayer):  ...")
    print("  supports_gradient_checkpointing = True   # on GlmMoeDsaPreTrainedModel")
    print("  self.gradient_checkpointing = False      # default in GlmMoeDsaModel.__init__")
    print("  trainer / model.gradient_checkpointing_enable()  # flip ON for training")

    print(f"\n{'─' * 72}")
    print("  Tradeoff: less memory, more compute (forward runs twice per checkpointed layer).")
    print("  Inference (.eval()): checkpointing is skipped even if flag is True.")
    print(f"{'─' * 72}\n")


if __name__ == "__main__":
    main()
