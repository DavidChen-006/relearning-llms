"""Row 4 test — GlmMoeDsaDecoderLayer (composes RMSNorm + Attention + MLP). AAA."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "architecture"))   # add david/architecture to the path

import torch
from transformers import GlmMoeDsaConfig

from modeling_glm_moe_dsa import GlmMoeDsaDecoderLayer, GlmMoeDsaRotaryEmbedding


def causal_mask(seq):
    return torch.full((seq, seq), float("-inf")).triu(1)


def test_decoder_layer():
    # ---------- ARRANGE ---------- (build the whole layer + its input)
    config = GlmMoeDsaConfig(hidden_size=8, num_attention_heads=2, intermediate_size=16)
    config._attn_implementation = "eager"
    layer = GlmMoeDsaDecoderLayer(config, layer_idx=0)
    torch.manual_seed(0)
    x = torch.randn(2, 4, config.hidden_size)      # (batch=2, seq=4, hidden=8)
    mask = causal_mask(4)
    # RoPE angles for positions 0..3 (the layer relays the (cos, sin) tuple to attention)
    rotary = GlmMoeDsaRotaryEmbedding(config)
    position_embeddings = rotary(x, torch.arange(4).unsqueeze(0))

    # ---------- ACT ---------- (run one full transformer layer)
    out = layer(x, mask, position_embeddings=position_embeddings)

    # ---------- ASSERT ---------- (the layer's properties)
    assert out.shape == x.shape, "layer must preserve shape (B, S, H) -> (B, S, H)"
    assert torch.isfinite(out).all(), "output must be all finite (no NaN/Inf)"

    # causal still holds through the whole layer (attention is causal; norm/MLP are per-token)
    x2 = x.clone()
    x2[:, -1, :] = torch.randn(2, config.hidden_size)   # perturb only the future token
    out2 = layer(x2, mask, position_embeddings=position_embeddings)
    assert torch.allclose(out[:, :-1, :], out2[:, :-1, :], atol=1e-5), "earlier tokens must not see the future"


if __name__ == "__main__":
    test_decoder_layer()
    print("PASS")
