"""Test embed_tokens: text → tokenizer → input_ids → vectors.

Run:
    python david/demo/demo_embed_tokens.py

Your hypothesis (with one fix):
  text  →  tokenizer  →  integer token IDs  →  embed_tokens  →  tensor of vectors
                                                          (not a Python list — a torch.Tensor)
"""

from __future__ import annotations

import torch
import torch.nn as nn

torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)

# Tiny char-level tokenizer (same idea as data.py / toy vocab_size=96)
VOCAB = sorted(set("abcdefghijklmnopqrstuvwxyz .,!?'"))
STOI = {ch: i for i, ch in enumerate(VOCAB)}
ITOS = {i: ch for i, ch in enumerate(VOCAB)}


def encode(text: str) -> torch.LongTensor:
    """text → 1D tensor of token IDs"""
    ids = [STOI[ch] for ch in text if ch in STOI]
    return torch.tensor(ids, dtype=torch.long)


def decode(ids: torch.LongTensor) -> str:
    return "".join(ITOS[int(i)] for i in ids)


class TinyModel(nn.Module):
    """Minimal stand-in for GlmMoeDsaModel — only the entry point."""

    def __init__(self, vocab_size: int, hidden_size: int, padding_idx: int = 0):
        super().__init__()
        self.padding_idx = padding_idx
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx)

    def embed_text(self, text: str) -> tuple[torch.LongTensor, torch.Tensor]:
        input_ids = encode(text).unsqueeze(0)  # (1, seq_len) — add batch dim
        vectors = self.embed_tokens(input_ids)  # (1, seq_len, hidden_size)
        return input_ids, vectors


def _header(title: str) -> None:
    print(f"\n{'─' * 72}\n  {title}\n{'─' * 72}")


def main() -> None:
    vocab_size = len(VOCAB)
    hidden_size = 8  # tiny so vectors print on one line

    model = TinyModel(vocab_size=vocab_size, hidden_size=hidden_size)
    print("TinyModel with self.embed_tokens = nn.Embedding(...)\n")
    print(f"  vocab_size={vocab_size}  hidden_size={hidden_size}")

    for text in ["hi", "hello", "a."]:
        _header(f'text = {text!r}')
        input_ids, vectors = model.embed_text(text)

        print(f"  1. tokenizer.encode → input_ids")
        print(f"     shape: {tuple(input_ids.shape)}   values: {input_ids[0].tolist()}")
        print(f"     chars: {[decode(input_ids[0, i : i + 1]) for i in range(input_ids.shape[1])]}")

        print(f"\n  2. self.embed_tokens(input_ids) → vectors")
        print(f"     shape: {tuple(vectors.shape)}  (batch, seq_len, hidden_size)")
        print(f"     vectors:\n{vectors[0].detach()}")

        print(f"\n  3. one position: token {input_ids[0, 0].item()!r} ({text[0]!r})")
        print(f"     vector = embed.weight[row]:")
        row = model.embed_tokens.weight[input_ids[0, 0]]
        print(f"     {row.detach().tolist()}")

    _header("Batch of two strings (padded manually)")
    t1, t2 = "hi", "hey"
    ids1, ids2 = encode(t1), encode(t2)
    max_len = max(len(ids1), len(ids2))
    pad_id = 0
    batch = torch.full((2, max_len), pad_id, dtype=torch.long)
    batch[0, : len(ids1)] = ids1
    batch[1, : len(ids2)] = ids2
    print(f"  input_ids batch:\n{batch}")
    batch_vectors = model.embed_tokens(batch)
    print(f"  vectors shape: {tuple(batch_vectors.shape)}")
    print(f"  vectors:\n{batch_vectors.detach()}")

    print(f"\n{'─' * 72}")
    print("  Pipeline:  text → encode() → input_ids → self.embed_tokens() → vectors")
    print("  In GLM:    text → HF tokenizer → input_ids → model.model.embed_tokens → layers...")
    print(f"{'─' * 72}\n")


if __name__ == "__main__":
    main()
