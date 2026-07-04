"""Row 2 test — GlmMoeDsaMLP alone. Arrange -> Act -> Assert."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "architecture"))   # add david/architecture to the path

import torch
from transformers import GlmMoeDsaConfig

from modeling_glm_moe_dsa import GlmMoeDsaMLP


def test_mlp():
    # ---------- ARRANGE ---------- (build the thing under test + its input)
    config = GlmMoeDsaConfig(hidden_size=8, intermediate_size=16, hidden_act="silu")
    mlp = GlmMoeDsaMLP(config)
    x = torch.randn(2, 3, config.hidden_size)   # (batch=2, seq=3, hidden=8)

    # ---------- ACT ---------- (run the one operation being tested)
    out = mlp(x)

    # ---------- ASSERT ---------- (check the properties that must hold)
    assert out.shape == x.shape, "MLP must preserve shape (B, S, H) -> (B, S, H)"
    assert torch.isfinite(out).all(), "output must be all finite (no NaN/Inf)"

    # per-token isolation: token 0's output must NOT depend on the other tokens
    out_solo = mlp(x[:, :1, :])                 # run ONLY token 0
    assert torch.allclose(out[:, :1, :], out_solo, atol=1e-6), "each token must be processed independently"


if __name__ == "__main__":
    test_mlp()
    print("PASS")
