"""demo_load_dataset.py — playground: what does load_dataset('json', ...) actually give you?

A/B test on the same tiny synthetic jsonl:
  A) plain python: read the file yourself with json.loads per line
  B) HF datasets:  load_dataset('json', data_files=..., split='train')  <- the line from
     lm_dataset.py's __init__: self.samples = load_dataset(...)
Then poke at what B can do that A can't.
"""
import json
import os

from datasets import load_dataset

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------- make a tiny synthetic jsonl (same {"text": ...} shape as our real data) ----------
docs = [
    {"text": "The cat sat on the mat."},
    {"text": "GLM is rebuilt rung by rung."},
    {"text": "Shakespeare wrote many plays."},
    {"text": "Tokenizers map text to ids."},
    {"text": "The loss should go down."},
]
with open("playground_data.jsonl", "w") as f:
    for d in docs:
        f.write(json.dumps(d) + "\n")

# ============================ A: plain python ============================
print("=" * 70)
print("A) plain python: json.loads per line")
print("=" * 70)
samples_a = [json.loads(line) for line in open("playground_data.jsonl")]
print("type:        ", type(samples_a))
print("len:         ", len(samples_a))
print("samples_a[1]:", samples_a[1])
print(" -> a plain list of dicts, fully loaded into RAM. That's it.")

# ============================ B: load_dataset ============================
print("\n" + "=" * 70)
print("B) load_dataset('json', data_files=..., split='train')")
print("=" * 70)
samples_b = load_dataset("json", data_files="playground_data.jsonl", split="train")
print("type:        ", type(samples_b))
print("len:         ", len(samples_b))
print("samples_b[1]:", samples_b[1])
print("schema:      ", samples_b.features)          # it inferred a typed schema from the json

# ---------- things the plain list can't do (or can't do this easily) ----------
print("\n--- column access: the whole 'text' column at once ---")
print(samples_b["text"])

print("\n--- slicing returns columns-of-lists, not list-of-dicts ---")
print("samples_b[1:3]:", samples_b[1:3])

print("\n--- map: transform every sample (batched, cached, multiprocess-able) ---")
upper = samples_b.map(lambda s: {"text": s["text"].upper()})
print("after .map(upper):", upper[0])

print("\n--- filter: keep a subset ---")
short = samples_b.filter(lambda s: len(s["text"]) < 26)
print(f"docs shorter than 26 chars: {len(short)} of {len(samples_b)} ->", short["text"])

print("\n--- train_test_split: the 90/10 split as a one-liner ---")
split = samples_b.train_test_split(test_size=0.4, seed=0)
print("train:", split["train"]["text"])
print("test: ", split["test"]["text"])

# ---------- the REAL reason training code uses it: memory ----------
print("\n" + "=" * 70)
print("WHY training code prefers B — memory-mapped Arrow, not RAM")
print("=" * 70)
print(samples_b.cache_files)
print("""
 -> load_dataset converted the jsonl ONCE into an Arrow file on disk (see cache
    path above) and MEMORY-MAPS it: samples are read from disk on access, so a
    100GB corpus works on a 16GB laptop. A plain python list (A) must fit
    entirely in RAM. Same [i] access for us either way — that's why swapping
    A<->B wouldn't change lm_dataset.py's logic, only its scaling ceiling.""")
