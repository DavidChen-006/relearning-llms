"""Row 6 test — GlmMoeDsaForCausalLM (Model + lm_head). ids -> logits = full inference. AAA."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "architecture"))   # add david/architecture to the path

import torch
from transformers import GlmMoeDsaConfig

from modeling_glm_moe_dsa import GlmMoeDsaForCausalLM


def test_causal_lm():
    # ---------- ARRANGE ---------- (build the whole model + token-id input)
    config = GlmMoeDsaConfig(
        vocab_size=32, hidden_size=8, num_attention_heads=2,
        intermediate_size=16, num_hidden_layers=2,
    )
    config._attn_implementation = "eager"
    model = GlmMoeDsaForCausalLM(config)
    model.config._attn_implementation = "eager"
    model.eval()
    torch.manual_seed(0)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))    # (batch=2, seq=5) token ids

    # ---------- ACT ---------- (ids -> logits: the full forward pass)
    with torch.no_grad():
        logits = model(input_ids)

    # ---------- ASSERT ---------- (properties of full inference)
    assert logits.shape == (2, 5, config.vocab_size), "ids (B, S) -> logits (B, S, vocab)"
    assert torch.isfinite(logits).all(), "output must be all finite (no NaN/Inf)"

    # inference works: the last token's argmax is a valid predicted token id
    next_id = logits[:, -1, :].argmax(-1)
    assert next_id.shape == (2,), "one predicted id per sequence in the batch"
    assert ((next_id >= 0) & (next_id < config.vocab_size)).all(), "predicted id must be a real token"

    # causal still holds end-to-end: change the last token, earlier logits unchanged
    ids2 = input_ids.clone()
    ids2[:, -1] = (input_ids[:, -1] + 1) % config.vocab_size
    with torch.no_grad():
        logits2 = model(ids2)
    assert torch.allclose(logits[:, :-1, :], logits2[:, :-1, :], atol=1e-5), "earlier tokens must not see the future"


if __name__ == "__main__":
    test_causal_lm()
    print("PASS")
