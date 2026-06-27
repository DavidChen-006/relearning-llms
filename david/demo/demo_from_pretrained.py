"""A/B: loading a checkpoint — manual (old way) vs from_pretrained (HF way).

Run:
    python david/demo/demo_from_pretrained.py

Historical context (why the class is named PreTrainedModel):
  In 2018–2019 the #1 job was NOT training BERT from scratch. It was:

      model = BertModel.from_pretrained("bert-base-uncased")

  Everyone downloaded a finished checkpoint. HuggingFace built PreTrainedModel
  so .from_pretrained() "just works" — config, weights, naming, device map, etc.

  The name means "this class supports the pretrained-checkpoint workflow",
  not "weights are already trained inside __init__".
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.init as init
from safetensors.torch import load_file
from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_utils import PreTrainedModel


class ToyConfig(PreTrainedConfig):
    model_type = "toy_demo"
    vocab_size: int = 32
    hidden_size: int = 16


class ToyForCausalLM(PreTrainedModel):
    config_class = ToyConfig
    config: ToyConfig

    def __init__(self, config: ToyConfig):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.lm_head(self.norm(self.embed(input_ids)))

    def _init_weights(self, module: nn.Module) -> None:
        super()._init_weights(module)
        if isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=0.02)


# ── A: old way — you wire every piece yourself ────────────────────────────────


def load_checkpoint_old_way(ckpt_dir: Path) -> ToyForCausalLM:
    """How you loaded models before HuggingFace unified the workflow."""
    # 1. Read hyperparameters from disk yourself
    with open(ckpt_dir / "config.json") as f:
        raw = json.load(f)
    config = ToyConfig(**raw)

    # 2. Build empty architecture (random weights — you'll overwrite them)
    model = ToyForCausalLM(config)

    # 3. Find the weight file yourself
    weights_path = ckpt_dir / "model.safetensors"

    # 4. Load raw tensors yourself
    state_dict = load_file(str(weights_path))

    # 5. Map keys into the model yourself (breaks if names don't match exactly)
    model.load_state_dict(state_dict)

    # 6. Set eval mode yourself (easy to forget)
    model.eval()
    return model


# ── B: HF way — one call, same result ─────────────────────────────────────────


def load_checkpoint_hf_way(ckpt_dir: Path) -> ToyForCausalLM:
    """What BertModel.from_pretrained("bert-base-uncased") does under the hood."""
    return ToyForCausalLM.from_pretrained(ckpt_dir)


# ── Demo helpers ──────────────────────────────────────────────────────────────


def _header(title: str) -> None:
    print(f"\n{'─' * 72}\n  {title}\n{'─' * 72}")


def _count_lines(fn) -> int:
    import inspect

    return len(inspect.getsource(fn).strip().splitlines())


def main() -> None:
    print("A/B: loading a saved checkpoint")
    print("A = manual steps (pre-HF workflow)  |  B = from_pretrained (why PreTrainedModel exists)\n")

    torch.manual_seed(0)
    config = ToyConfig(vocab_size=32, hidden_size=16)
    input_ids = torch.tensor([[1, 2, 3]])

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "toy-demo"

        # Teacher model — pretend this was downloaded from the Hub as "bert-base-uncased"
        teacher = ToyForCausalLM(config)
        teacher.save_pretrained(ckpt)
        with torch.no_grad():
            teacher_logits = teacher(input_ids)

        print(f"  Checkpoint on disk: {sorted(p.name for p in ckpt.iterdir())}")
        print('  (Imagine config.json + model.safetensors came from "bert-base-uncased")\n')

        _header("A  OLD WAY — 6 manual steps you used to write yourself")
        print(f"  Source lines in load_checkpoint_old_way: {_count_lines(load_checkpoint_old_way)}")
        print("    1. open(config.json) + parse JSON")
        print("    2. build model from config")
        print("    3. find weight file on disk")
        print("    4. load tensors (safetensors or torch.load)")
        print("    5. model.load_state_dict(...)  ← key mismatch = cryptic error")
        print("    6. model.eval()")

        t0 = time.perf_counter()
        model_a = load_checkpoint_old_way(ckpt)
        time_a = time.perf_counter() - t0
        with torch.no_grad():
            logits_a = model_a(input_ids)
        diff_a = (teacher_logits - logits_a).abs().max().item()
        print(f"\n  loaded in {time_a * 1000:.1f} ms")
        print(f"  max |teacher - A| = {diff_a:.2e}  {'OK' if diff_a == 0.0 else 'MISMATCH'}")

        _header('B  HF WAY — what BertModel.from_pretrained("bert-base-uncased") looks like')
        print("    model = ToyForCausalLM.from_pretrained(ckpt_dir)")
        print(f"  (PreTrainedModel handles steps 1–6 + dtype/device/key remapping)\n")

        t0 = time.perf_counter()
        model_b = load_checkpoint_hf_way(ckpt)
        time_b = time.perf_counter() - t0
        with torch.no_grad():
            logits_b = model_b(input_ids)
        diff_b = (teacher_logits - logits_b).abs().max().item()
        print(f"  loaded in {time_b * 1000:.1f} ms")
        print(f"  max |teacher - B| = {diff_b:.2e}  {'OK' if diff_b == 0.0 else 'MISMATCH'}")

        _header("Same weights, same logits — different amount of glue code")
        print(f"  A logits mean: {logits_a.mean():.6f}")
        print(f"  B logits mean: {logits_b.mean():.6f}")
        print(f"  A == B: {torch.allclose(logits_a, logits_b)}")

    print(f"\n{'─' * 72}")
    print('  Historical use case:  model = BertModel.from_pretrained("bert-base-uncased")')
    print("  PreTrainedModel = the base class that makes .from_pretrained() possible.")
    print("  It does NOT mean weights are pretrained when you call Model(config).")
    print("  It means: this architecture knows how to load a finished checkpoint.")
    print(f"{'─' * 72}\n")


if __name__ == "__main__":
    main()
