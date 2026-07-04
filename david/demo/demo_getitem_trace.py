"""demo_getitem_trace.py — ONE document's journey through PretrainDataset.__getitem__.

Re-enacts each line of lm_dataset.py's __getitem__ on one real sample from the
Shakespeare jsonl, printing every intermediate state. The five stages:
    FETCH -> TOKENIZE -> FRAME -> PAD -> GRADE
"""
import os

import torch
from transformers import AutoTokenizer

from datasets import load_dataset

os.chdir(os.path.dirname(os.path.abspath(__file__)))

MAX_LENGTH = 64          # small so every token fits on screen (real code: 512)
INDEX = 1                # which document to trace

tok = AutoTokenizer.from_pretrained("..")
samples = load_dataset("json", data_files="../training/data/pretrain_shakespeare.jsonl", split="train")

print("=" * 72)
print(f"STAGE 1 — FETCH:  sample = self.samples[{INDEX}]")
print("=" * 72)
sample = samples[INDEX]
print("raw record:", sample)

print("\n" + "=" * 72)
print(f"STAGE 2 — TOKENIZE:  tokenizer(text, max_length={MAX_LENGTH}-2, truncation=True)")
print("=" * 72)
tokens = tok(str(sample["text"]), add_special_tokens=False,
             max_length=MAX_LENGTH - 2, truncation=True).input_ids
print(f"{len(tokens)} ids:", tokens)
print("id -> chunk:", [(i, tok.decode([i])) for i in tokens])
print(" -> note max_length-2: two seats are being saved for the next stage")

print("\n" + "=" * 72)
print("STAGE 3 — FRAME:  [bos] + tokens + [eos]")
print("=" * 72)
tokens = [tok.bos_token_id] + tokens + [tok.eos_token_id]
print(f"{len(tokens)} ids:", tokens)
print(f" -> {tok.bos_token_id} = {tok.bos_token!r} in front, {tok.eos_token_id} = {tok.eos_token!r} at the end")
print("    (every training doc ends with EOS -> the model LEARNS when to stop)")

print("\n" + "=" * 72)
print(f"STAGE 4 — PAD:  + [pad] * ({MAX_LENGTH} - {len(tokens)})")
print("=" * 72)
input_ids = tokens + [tok.pad_token_id] * (MAX_LENGTH - len(tokens))
input_ids = torch.tensor(input_ids, dtype=torch.long)
print(f"{len(input_ids)} ids:", input_ids.tolist())
print(f" -> rectangle achieved: every sample is exactly {MAX_LENGTH} long, so batches stack")

print("\n" + "=" * 72)
print("STAGE 5 — GRADE:  labels = clone; labels[pad] = -100")
print("=" * 72)
labels = input_ids.clone()
labels[input_ids == tok.pad_token_id] = -100
print("labels:", labels.tolist())

print("\n" + "=" * 72)
print("FINAL — what the training loop receives, position by position")
print("=" * 72)
print(f"{'pos':>4} {'input_id':>9} {'token':<18} {'label':>7}  graded?")
for p in range(MAX_LENGTH):
    i, lab = input_ids[p].item(), labels[p].item()
    name = repr(tok.decode([i]))
    mark = "yes" if lab != -100 else "NO (-100)"
    print(f"{p:>4} {i:>9} {name:<18} {lab:>7}  {mark}")
    if lab == -100 and p > 0 and labels[p - 1].item() == -100 and p < MAX_LENGTH - 1:
        print(f"     ... (padding continues to position {MAX_LENGTH - 1}) ...")
        break

print("\nremember: labels are UNSHIFTED here (aligned with input_ids).")
print("the shift to 'predict the NEXT token' happens later, in the loss —")
print("that was the design fork: minimind shifts in its model, train.py in compute_loss.")
