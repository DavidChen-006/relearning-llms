"""Row 3 test — GlmMoeDsaAttention alone. Arrange -> Act -> Assert."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "architecture"))   # add david/architecture to the path

import torch
from transformers import GlmMoeDsaConfig

from modeling_glm_moe_dsa import GlmMoeDsaAttention, GlmMoeDsaRotaryEmbedding


def causal_mask(seq):   # -inf above the diagonal so a token can't see the future
    return torch.full((seq, seq), float("-inf")).triu(1)


def test_attention():
    # ---------- ARRANGE ---------- (build the thing under test + its input)
    config = GlmMoeDsaConfig(hidden_size=8, num_attention_heads=2)
    config._attn_implementation = "eager"          # use our plain softmax(QKt)*V fallback
    attn = GlmMoeDsaAttention(config, layer_idx=0)
    torch.manual_seed(0)
    x = torch.randn(2, 4, config.hidden_size)      # (batch=2, seq=4, hidden=8)
    mask = causal_mask(4)
    # RoPE angles for positions 0..3 (attention needs the (cos, sin) tuple)
    rotary = GlmMoeDsaRotaryEmbedding(config)
    position_embeddings = rotary(x, torch.arange(4).unsqueeze(0))

    # ---------- ACT ---------- (run the one operation being tested)
    out, _weights = attn(x, position_embeddings, mask)

    # ---------- ASSERT ---------- (check the properties that must hold)
    assert out.shape == x.shape, "attention must preserve shape (B, S, H) -> (B, S, H)"
    assert torch.isfinite(out).all(), "output must be all finite (no NaN/Inf)"

    # causal: changing the LAST token must NOT change the earlier tokens' outputs
    x2 = x.clone()
    x2[:, -1, :] = torch.randn(2, config.hidden_size)   # perturb only the future token
    out2, _ = attn(x2, position_embeddings, mask)
    assert torch.allclose(out[:, :-1, :], out2[:, :-1, :], atol=1e-5), "earlier tokens must not see the future"


if __name__ == "__main__":
    test_attention()
    print("PASS")
