"""prepare_data.py — encode raw Shakespeare into training-ready id files, using a real
BPE tokenizer (minimind's 6400-vocab tokenizer.json, HF format — same layout as GLM's).

Mirrors the real package split:
  tokenizer.json          <- the tokenizer (portable JSON, lives in the repo)
  data/train.bin          <- encoded ids, first 90%   (uint16 binary, regenerable)
  data/val.bin            <- encoded ids, last 10%
"""
import os

import numpy as np
from transformers import AutoTokenizer

os.chdir(os.path.dirname(os.path.abspath(__file__)))     # paths relative to training/, wherever run from

text = open("data/tinyshakespeare.txt").read()

tokenizer = AutoTokenizer.from_pretrained("..")          # reads tokenizer.json in david/ (next to inference.py)
ids = tokenizer(text)["input_ids"]                       # BPE-encode the whole corpus

n = int(0.9 * len(ids))                                  # 90/10: val = unseen text for honest loss
train_ids = np.array(ids[:n], dtype=np.uint16)           # uint16 fits vocab 6400 (max 65535)
val_ids = np.array(ids[n:], dtype=np.uint16)
train_ids.tofile("data/train.bin")
val_ids.tofile("data/val.bin")

if __name__ == "__main__":
    print(f"vocab_size: {tokenizer.vocab_size}")
    print(f"chars in:   {len(text):,}")
    print(f"tokens out: {len(ids):,}   train: {len(train_ids):,}   val: {len(val_ids):,}")
    print(f"compression: {len(text) / len(ids):.2f} chars/token")
    sample_ids = train_ids[:12].tolist()
    print(f"round-trip: {sample_ids} -> {tokenizer.decode(sample_ids)!r}")
