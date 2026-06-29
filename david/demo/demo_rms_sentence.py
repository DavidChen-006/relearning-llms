"""demo_rms_sentence.py — RMSNorm on a real sentence, in 2 sections x 3 parts.

Parts (both sections):
  1. PRE-RMS                      (raw embedding)
  2. POST-RMS, no trained weight  (x / RMS,  gamma = 1)
  3. RMSNorm WITH trained weight  (x / RMS * gamma)   <- gamma = GPT-2's real ln_f weight
Sections:
  1. VISUALIZATION (heatmaps)
  2. ONE TOKEN     (actual numbers)
"""
from transformers import AutoModelForCausalLM, AutoTokenizer


def rms_norm_noweight(x, eps=1e-5):
    return x / x.pow(2).mean(-1, keepdim=True).add(eps).sqrt()


def _cell(r, g, b):
    return f"\x1b[48;2;{int(r)};{int(g)};{int(b)}m \x1b[0m"


def heatmap(label, mat, ncols=96):
    m = mat.detach()[:, :ncols]
    scale = m.abs().max().item() + 1e-9
    print(f"\n{label}  (first {ncols} of {mat.shape[1]} dims):")
    for t in range(m.shape[0]):
        row = ""
        for d in range(m.shape[1]):
            v = max(-1.0, min(1.0, float(m[t, d]) / scale))
            row += _cell(255, 255 * (1 - v), 255 * (1 - v)) if v >= 0 else _cell(255 * (1 + v), 255 * (1 + v), 255)
        print(f"  tok{t} " + row)


def rms_of(vec):
    return round(float(vec.pow(2).mean().sqrt()), 3)


tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

sentence = "The cat sat on the mat"
ids = tok(sentence, return_tensors="pt").input_ids
embed = model.get_input_embeddings()(ids)[0]            # (tokens, 768) real embeddings
tokens = [tok.decode(i).strip() or "_" for i in ids[0]]
gamma = model.transformer.ln_f.weight.detach()          # real trained per-feature weight (768)

pre = embed                                             # part 1
post = rms_norm_noweight(embed)                         # part 2
post_w = post * gamma                                   # part 3

print("sentence:", sentence, "| tokens:", tokens, "| shape:", tuple(embed.shape))

# ================= SECTION 1: VISUALIZATION =================
print("\n" + "#" * 60)
print("# SECTION 1 — VISUALIZATION (heatmaps)")
print("#" * 60)
heatmap("1) PRE-RMS  (raw embedding)", pre)
heatmap("2) POST-RMS (no trained weight, gamma=1) — rows rescaled to RMS 1", post)
heatmap("3) RMSNorm WITH trained weight (x/RMS * gamma) — gamma reshapes per-feature", post_w)

# ================= SECTION 2: ONE TOKEN =================
ti, n = 1, 12     # token 1 = "cat", first 12 features
print("\n" + "#" * 60)
print(f"# SECTION 2 — ONE TOKEN: '{tokens[ti]}'  (first {n} of 768 features)")
print("#" * 60)
print("1) PRE-RMS    :", [round(x, 3) for x in pre[ti, :n].tolist()], "  RMS =", rms_of(pre[ti]))
print("2) POST-RMS   :", [round(x, 3) for x in post[ti, :n].tolist()], "  RMS =", rms_of(post[ti]))
print("3) WITH gamma :", [round(x, 3) for x in post_w[ti, :n].tolist()], "  RMS =", rms_of(post_w[ti]))
print("   gamma[:n]  :", [round(x, 3) for x in gamma[:n].tolist()], "  <- per-feature trained scale")
print("\nPart 2 = part 1 / its RMS (size -> 1).  Part 3 = part 2 * gamma (each feature re-scaled).")
