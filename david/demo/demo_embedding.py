"""nn.Embedding step-by-step — lookup table, not matrix multiply.

Run:
    python david/demo/demo_embedding.py

  token ID in  →  pick row from weight table  →  vector out
  (no @ weight.T — that's nn.Linear)
"""

from __future__ import annotations

import torch
import torch.nn as nn

torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)

VOCAB_SIZE = 5
HIDDEN_SIZE = 3


def _header(title: str) -> None:
    print(f"\n{'─' * 72}\n  {title}\n{'─' * 72}")


def _show(name: str, t: torch.Tensor) -> None:
    print(f"\n  {name}")
    print(f"    shape: {tuple(t.shape)}")
    print(f"    values:\n{t}")


def make_embed() -> nn.Embedding:
    embed = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
    # Fixed table so you can follow lookups by hand
    with torch.no_grad():
        embed.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],  # row 0 = token 0
                    [0.0, 1.0, 0.0],  # row 1 = token 1
                    [0.0, 0.0, 1.0],  # row 2 = token 2
                    [1.0, 1.0, 0.0],  # row 3 = token 3
                    [0.5, 0.5, 0.5],  # row 4 = token 4
                ]
            )
        )
    return embed


def part1_single_ids() -> None:
    _header("Part 1 — one token ID  input shape ()")

    embed = make_embed()
    token_id = torch.tensor(2)

    _show("STEP 0 — token ID (integer)", token_id)
    _show("STEP 1 — weight table  shape (vocab_size, hidden_size)", embed.weight)
    print("\n  Each row IS the vector for that token ID.")

    row = embed.weight[token_id]
    _show(f"STEP 2 — manual lookup  weight[{token_id.item()}]", row)

    out = embed(token_id)
    _show("STEP 3 — embed(token_id)", out)
    print(f"\n  match: {torch.equal(row, out)}")


def part2_sequence() -> None:
    _header("Part 2 — sequence of IDs  input shape (seq_len,)")

    embed = make_embed()
    input_ids = torch.tensor([2, 0, 4])  # three tokens

    _show("STEP 0 — input_ids", input_ids)
    _show("STEP 1 — weight table", embed.weight)

    print("\n  Lookup each ID → stack rows:")
    for i, tid in enumerate(input_ids):
        print(f"    position {i}: id={tid.item()}  →  {embed.weight[tid].tolist()}")

    out = embed(input_ids)
    _show("STEP 2 — embed(input_ids)  shape (seq_len, hidden_size)", out)


def part3_batch() -> None:
    _header("Part 3 — batch + sequence  input shape (batch, seq_len)")

    embed = make_embed()
    input_ids = torch.tensor(
        [
            [2, 0],  # batch 0
            [4, 3],  # batch 1
        ]
    )

    _show("STEP 0 — input_ids", input_ids)
    out = embed(input_ids)
    _show("STEP 1 — embed(input_ids)  shape (batch, seq_len, hidden_size)", out)

    print("\n  Zoom batch 1, position 0: id=4")
    _show("  weight[4]", embed.weight[4])
    _show("  output[1, 0]", out[1, 0])
    print(f"  match: {torch.equal(embed.weight[4], out[1, 0])}")


def part4_vs_linear() -> None:
    _header("Part 4 — Embedding vs Linear (why GLM uses Embedding at the door)")

    print("  nn.Embedding  →  input must be integers (token IDs)")
    print("  nn.Linear     →  input must be floats (vectors)")
    print()
    embed = make_embed()
    linear = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)

    ids = torch.tensor([2, 0])
    vectors = embed(ids)
    _show("after embed — float vectors", vectors)

    logits = linear(vectors)
    _show("after linear — logits per position", logits)
    print("\n  GLM path:  input_ids → embed_tokens → layers → norm → lm_head")


def part5_padding() -> None:
    _header("Part 5 — padding_idx (optional)")

    pad_id = 0
    embed = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE, padding_idx=pad_id)
    with torch.no_grad():
        embed.weight.copy_(torch.randn(VOCAB_SIZE, HIDDEN_SIZE))
        embed.weight[pad_id] = 99.0  # try to poison pad row

    # post_init-style zeroing happens in HF _init_weights; here we show the rule:
    with torch.no_grad():
        embed.weight[pad_id].zero_()

    ids = torch.tensor([2, 0, 4])  # 0 is padding
    out = embed(ids)
    print(f"  padding_idx={pad_id}  →  row {pad_id} forced to zeros")
    _show("embed([2, 0, 4])", out)
    print(f"  output at pad position is zero: {torch.all(out[1] == 0).item()}")


def main() -> None:
    print("nn.Embedding — lookup table for token IDs\n")
    print("  Formula:  output = weight[token_id]   (row pick, not matmul)")

    part1_single_ids()
    part2_sequence()
    part3_batch()
    part4_vs_linear()
    part5_padding()

    print(f"\n{'─' * 72}")
    print("  embed_tokens in GlmMoeDsaModel = nn.Embedding(vocab_size, hidden_size, padding_idx)")
    print("  weight shape: (vocab_size, hidden_size) — one row per token in the vocabulary")
    print(f"{'─' * 72}\n")


if __name__ == "__main__":
    main()
