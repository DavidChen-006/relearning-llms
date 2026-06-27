"""Rung 0 — data: char-level tokenizer + batching.

The model only does math on numbers, so we map every unique character to an
integer (the tokenizer). Then get_batch() serves random (input, target) pairs
where the target is the input shifted one step right — i.e. "predict the next char".
"""
import torch

# ---- load the raw text ----
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# ---- build the vocabulary: every unique character ----
chars = sorted(set(text))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}   # char  -> integer id
itos = {i: ch for i, ch in enumerate(chars)}   # integer id -> char

def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]

def decode(ids: list[int]) -> str:
    return "".join(itos[i] for i in ids)

# ---- encode the whole dataset once, and split 90/10 into train/val ----
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split: str, batch_size: int, block_size: int, device: str):
    """Return one batch of (x, y).

    x : (batch_size, block_size)  -- chunks of text
    y : (batch_size, block_size)  -- the SAME chunks shifted right by 1 (the targets)
    """
    src = train_data if split == "train" else val_data
    # pick batch_size random start positions
    ix = torch.randint(len(src) - block_size, (batch_size,))
    x = torch.stack([src[i : i + block_size] for i in ix])
    y = torch.stack([src[i + 1 : i + 1 + block_size] for i in ix])  # <-- shifted by one
    return x.to(device), y.to(device)
