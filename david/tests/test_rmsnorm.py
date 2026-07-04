"""Row 1 test — GlmMoeDsaRMSNorm alone. Arrange -> Act -> Assert."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "architecture"))   # add david/architecture to the path

import torch
from modeling_glm_moe_dsa import GlmMoeDsaRMSNorm


def per_row_rms(t):   # helper: the "size" of each token row
    return [round(r, 3) for r in t.pow(2).mean(-1).sqrt().flatten().tolist()]


def test_rmsnorm():
    # ---------- ARRANGE ---------- (build the thing under test + its input)
    H = 8
    norm = GlmMoeDsaRMSNorm(H)
    x = torch.randn(2, 3, H) * 5           # (batch=2, seq=3, hidden=8), deliberately varied scale

    # ---------- ACT ---------- (run the one operation being tested)
    out = norm(x)

    # ---------- ASSERT ---------- (check the properties that must hold)
    print("per-row RMS before:", per_row_rms(x))
    print("per-row RMS after :", per_row_rms(out))
    assert out.shape == x.shape, "shape must be preserved"
    assert all(abs(r - 1.0) < 1e-3 for r in per_row_rms(out)), "each row should be RMS 1"


if __name__ == "__main__":
    test_rmsnorm()
    print("PASS")
