"""Row 5 test — GlmMoeDsaModel (embed + N layers + norm). ids -> hidden states. AAA."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "architecture"))   # add david/architecture to the path

import torch
from transformers import GlmMoeDsaConfig

from modeling_glm_moe_dsa import GlmMoeDsaModel


def test_model():
    # ---------- ARRANGE ---------- (build the whole body + token-id input)
    config = GlmMoeDsaConfig(
        vocab_size=32, hidden_size=8, num_attention_heads=2,
        intermediate_size=16, num_hidden_layers=2,
    )
    config._attn_implementation = "eager"
    model = GlmMoeDsaModel(config)
    model.config._attn_implementation = "eager"
    model.eval()
    torch.manual_seed(0)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))    # (batch=2, seq=5) token ids

    # ---------- ACT ---------- (ids -> hidden states)
    with torch.no_grad():
        hidden = model(input_ids)

    # ---------- ASSERT ---------- (properties of the body)
    assert hidden.shape == (2, 5, config.hidden_size), "ids (B, S) -> hidden (B, S, H)"
    assert torch.isfinite(hidden).all(), "output must be all finite (no NaN/Inf)"

    # causal still holds across the whole stack: change the last token, earlier states unchanged
    ids2 = input_ids.clone()
    ids2[:, -1] = (input_ids[:, -1] + 1) % config.vocab_size   # different last token
    with torch.no_grad():
        hidden2 = model(ids2)
    assert torch.allclose(hidden[:, :-1, :], hidden2[:, :-1, :], atol=1e-5), "earlier tokens must not see the future"


if __name__ == "__main__":
    test_model()
    print("PASS")
