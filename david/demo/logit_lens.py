"""logit_lens.py — run it and watch."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "gpt2"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)
model.eval()

prompt = "The Eiffel Tower is located in the city of"
ids = tok(prompt, return_tensors="pt").input_ids

with torch.no_grad():
    out = model(ids, output_hidden_states=True)

ln_f = model.transformer.ln_f          # the model's final norm
hiddens = out.hidden_states            # residual stream after each layer (+ the embedding)

print(f'\nprompt: "{prompt} ___"\n')
print(f"{'layer':>6} | {'current guess':<16} | confidence")
print("-" * 42)
for i, h in enumerate(hiddens):
    h_last = h[:, -1, :]                       # last token's slice of the residual stream
    logits = model.lm_head(ln_f(h_last))       # the lens: final-norm + unembed
    p, idx = logits.softmax(-1).max(-1)
    label = "embed" if i == 0 else f"{i}"
    print(f"{label:>6} | {tok.decode(idx[0])!r:<16} | {p.item():4.0%}")
print()
