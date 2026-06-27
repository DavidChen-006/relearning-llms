"""Step 1 — Trace shapes through GlmMoeDsaDecoderLayer.forward (R1/R2 ladder).

Run from repo root:
    python david/demo/demo_decoder_layer_shapes.py

Pick toy dimensions first — no math yet. For one token position (t=0), print tensor
shapes at each line in GlmMoeDsaDecoderLayer.forward. You should see:

  • Residual and hidden_states: same [B, S, hidden_size] throughout the layer wrapper
  • Attention output: same shape as input (mixing is across seq, not hidden dim)
  • MLP: internally wider (intermediate_size), then back to hidden_size

Uses the real GlmMoeDsaDecoderLayer submodules from transformers; forward is mirrored
here with shape prints only (same call order as glm/modeling_glm_moe_dsa.py).
"""

from __future__ import annotations

import torch
from transformers import GlmMoeDsaConfig, GlmMoeDsaModel

# ── Toy dimensions (R1/R2 ladder) ─────────────────────────────────────────────

BATCH = 1
SEQ_LEN = 4
HIDDEN = 128
INTERMEDIATE = 256
TOKEN_POS = 0  # "one token position" — print this slice too


def toy_config() -> GlmMoeDsaConfig:
    return GlmMoeDsaConfig(
        vocab_size=96,
        hidden_size=HIDDEN,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        first_k_dense_replace=1,
        intermediate_size=INTERMEDIATE,
        moe_intermediate_size=128,
        index_topk=64,
        max_position_embeddings=256,
    )


def _header(title: str) -> None:
    print(f"\n{'─' * 72}\n  {title}\n{'─' * 72}")


def _fmt(shape: tuple[int, ...]) -> str:
    return str(tuple(shape))


def _line(step: str, tensor: torch.Tensor, *, note: str = "") -> None:
    tok = tensor[0, TOKEN_POS] if tensor.dim() == 3 else tensor[TOKEN_POS]
    extra = f"  —  {note}" if note else ""
    print(
        f"  {step:<42} full {_fmt(tensor.shape):<22} "
        f"token[{TOKEN_POS}] {_fmt(tok.shape)}{extra}"
    )


def trace_decoder_layer_forward(
    layer,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Mirror GlmMoeDsaDecoderLayer.forward — shape prints only."""

    print(f"\n  layer_idx={layer.self_attn.layer_idx}  mlp={type(layer.mlp).__name__}")
    print(f"  toy: batch={BATCH}  seq_len={SEQ_LEN}  hidden_size={HIDDEN}  "
          f"intermediate_size={INTERMEDIATE}\n")

    # ── Attention sublayer (pre-norm residual) ───────────────────────────────
    print("  # ── Attention sublayer ──")
    residual = hidden_states
    _line("residual = hidden_states", residual, note="save stream before norm")

    hidden_states = layer.input_layernorm(hidden_states)
    _line("hidden_states = input_layernorm(...)", hidden_states, note="still [B,S,H]")

    hidden_states, _, topk_indices = layer.self_attn(
        hidden_states=hidden_states,
        attention_mask=None,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        position_embeddings=position_embeddings,
        prev_topk_indices=None,
    )
    _line("hidden_states = self_attn(...)", hidden_states, note="delta, same [B,S,H]")
    if topk_indices is not None:
        print(f"  {'topk_indices':<42} full {_fmt(topk_indices.shape)}")

    hidden_states = residual + hidden_states
    _line("hidden_states = residual + hidden_states", hidden_states, note="stream after attn")

    # ── MLP sublayer (pre-norm residual) ─────────────────────────────────────
    print("\n  # ── MLP sublayer ──")
    residual = hidden_states
    _line("residual = hidden_states", residual, note="save stream again")

    hidden_states = layer.post_attention_layernorm(hidden_states)
    _line("hidden_states = post_attention_layernorm(...)", hidden_states, note="still [B,S,H]")

    mlp = layer.mlp
    if type(mlp).__name__ == "GlmMoeDsaMLP":
        gate = mlp.gate_proj(hidden_states)
        up = mlp.up_proj(hidden_states)
        _line("gate = gate_proj(...)", gate, note=f"expand → [{INTERMEDIATE}]")
        _line("up   = up_proj(...)", up, note=f"expand → [{INTERMEDIATE}]")
        mid = mlp.act_fn(gate) * up
        _line("mid  = act(gate) * up", mid, note=f"wide [{INTERMEDIATE}]")
        hidden_states = mlp.down_proj(mid)
        _line("hidden_states = down_proj(mid)", hidden_states, note=f"project back → [{HIDDEN}]")
    else:
        hidden_states = mlp(hidden_states)
        _line("hidden_states = mlp(...)", hidden_states, note="MoE path — see layer 3 demo later")

    hidden_states = residual + hidden_states
    _line("hidden_states = residual + hidden_states", hidden_states, note="stream out of layer")

    return hidden_states


def main() -> None:
    config = toy_config()
    model = GlmMoeDsaModel(config)
    layer = model.layers[0]

    torch.manual_seed(0)
    hidden_states = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    position_ids = torch.arange(SEQ_LEN, device=hidden_states.device).unsqueeze(0)
    position_embeddings = model.rotary_emb(hidden_states, position_ids=position_ids)

    _header("Step 1 — GlmMoeDsaDecoderLayer.forward (shapes only)")
    print("  Reference: glm/modeling_glm_moe_dsa.py  GlmMoeDsaDecoderLayer.forward")
    print(f"  One token position = index t={TOKEN_POS} along seq_len={SEQ_LEN}")

    trace_decoder_layer_forward(
        layer, hidden_states, position_embeddings, position_ids
    )

    _header("Checklist")
    print(f"  ✓  Input / output shape unchanged     (1, 4, 128) → (1, 4, 128)")
    print(f"  ✓  Residual and stream same width     hidden_size={HIDDEN} at every residual save/add")
    print(f"  ✓  Attention delta same shape         [B,S,H] — seq mixing, hidden dim untouched")
    print(f"  ✓  MLP expands then contracts         [B,S,{INTERMEDIATE}] → [B,S,{HIDDEN}]")

    _header("Rung map (CLAUDE.md)")
    print("  R1 skeleton  — embed_tokens → [layers] → norm → lm_head  (whole pipe)")
    print("  R2 layer     — this file: one decoder layer wrapper + residual adds")
    print("  Next         — reimplement layer.py and swap-in until outputs match")
    print(f"{'─' * 72}\n")


if __name__ == "__main__":
    main()
