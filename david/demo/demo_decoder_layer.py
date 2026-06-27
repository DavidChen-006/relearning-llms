"""demo_decoder_layer.py — trace + VISUALIZE the residual stream through one GLM-MoE-DSA decoder layer.

The forward() WIRING is copied verbatim from the real GlmMoeDsaDecoderLayer. Additions:
  - trace()   : prints shape/mean/std at each step
  - heatmaps  : renders the 128 dims as colored cells so you SEE the change (no 640 numbers)

NOTE on logit lens: logit lens decodes the stream into *words* via the model's trained
unembedding. This layer has random weights and no unembedding, so logit lens here = noise.
The right tool for an untrained layer is the heatmap below (brightness = size of change).

Sub-modules:
  input_layernorm / post_attention_layernorm : REAL  GlmMoeDsaRMSNorm
  mlp                                         : REAL  GlmMoeDsaMLP  (dense SwiGLU)
  self_attn                                   : STUB  (real GlmMoeDsaAttention deferred)
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GlmMoeDsaConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaMLP, GlmMoeDsaRMSNorm


def trace(label, x):
    print(f"{label:<38} shape={tuple(x.shape)}  mean={x.mean().item():+.4f}  std={x.std().item():.4f}")


def _cell(r, g, b):
    return f"\x1b[48;2;{int(r)};{int(g)};{int(b)}m \x1b[0m"


def heatmap(label, mat, diverging):
    """Render a (tokens, dims) tensor as colored cells — one cell per dimension."""
    m = mat.detach().squeeze(0)                       # (1,T,D) -> (T,D)
    scale = m.abs().max().item() + 1e-9               # normalize per-map so structure is visible
    print(f"\n{label}")
    for t in range(m.shape[0]):
        row = []
        for d in range(m.shape[1]):
            v = max(-1.0, min(1.0, float(m[t, d]) / scale))
            if diverging:                             # blue(-) .. white(0) .. red(+)
                row.append(_cell(255, 255 * (1 - v), 255 * (1 - v)) if v >= 0
                           else _cell(255 * (1 + v), 255 * (1 + v), 255))
            else:                                     # magnitude glow: dark(small) -> bright(big)
                a = abs(v)
                row.append(_cell(255 * a, 165 * a, 30 * a))
        print(f"  tok{t} " + "".join(row))


def attn_pattern(label, attn, tokens):
    """One head's (query x key) attention matrix. Rows = QUERY (who looks),
    cols = KEY (looked at). Row-normalized; brighter = more attention."""
    a = attn.detach()
    n = a.shape[0]
    print(f"\n{label}")
    print(" " * 14 + "".join(f"{k:>2} " for k in range(n)) + "  <- key idx")
    for q in range(n):
        rmax = a[q].max().item() + 1e-9
        cells = ""
        for k in range(n):
            w = a[q, k].item() / rmax
            blk = _cell(255 * w, 165 * w, 30 * w)
            cells += blk + blk + " "
        print(f"  {tokens[q][:9]:>9} | {cells}")


class AttnStub(nn.Module):
    """Stand-in for GlmMoeDsaAttention (deferred). Shape-preserving 3-tuple output."""

    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, **kwargs):
        return self.proj(hidden_states), None, None


class DemoDecoderLayer(nn.Module):
    """Faithful copy of GlmMoeDsaDecoderLayer. forward() wiring is verbatim; it also stashes
    the key tensors in self.viz so we can draw heatmaps after."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = AttnStub(config.hidden_size)                              # STUB
        self.mlp = GlmMoeDsaMLP(config)                                            # REAL
        self.input_layernorm = GlmMoeDsaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = GlmMoeDsaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.viz = {}

    def forward(self, hidden_states):
        trace("0. input  (residual stream in)", hidden_states)
        self.viz["stream_in"] = hidden_states

        # ===== Sub-layer 1: Attention =====
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, _ = self.self_attn(hidden_states=hidden_states)
        self.viz["attn_update"] = hidden_states                   # the change attention adds
        trace("2. self_attn update", hidden_states)
        hidden_states = residual + hidden_states                  # <- RESIDUAL ADD #1
        trace("3. after residual add #1", hidden_states)

        # ===== Sub-layer 2: MLP =====
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        self.viz["mlp_update"] = hidden_states                    # the change the MLP adds
        trace("5. mlp update", hidden_states)
        hidden_states = residual + hidden_states                  # <- RESIDUAL ADD #2
        trace("6. output (residual stream out)", hidden_states)
        self.viz["stream_out"] = hidden_states
        return hidden_states


if __name__ == "__main__":
    config = GlmMoeDsaConfig(
        vocab_size=96, hidden_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=4,
        intermediate_size=256, moe_intermediate_size=128,
        n_routed_experts=8, num_experts_per_tok=2, first_k_dense_replace=1,
        max_position_embeddings=256, index_topk=64,
    )
    torch.manual_seed(0)
    layer = DemoDecoderLayer(config)

    x = torch.randn(1, 5, config.hidden_size)   # (batch=1, seq=5 tokens, hidden=128)
    print("Residual stream through ONE decoder layer  (5 tokens x 128 dims):\n")
    layer(x)

    print("\n" + "=" * 60)
    print("HEATMAPS — each cell = 1 of the 128 dims, per token row")
    print("  streams: blue = negative, red = positive")
    print("  updates: bright = big change at that dim, dark = ~no change")
    print("=" * 60)
    v = layer.viz
    heatmap("RESIDUAL STREAM IN", v["stream_in"], diverging=True)
    heatmap("  -> attention's UPDATE (what it writes into the stream)", v["attn_update"], diverging=False)
    heatmap("  -> MLP's UPDATE (what it writes into the stream)", v["mlp_update"], diverging=False)
    heatmap("RESIDUAL STREAM OUT", v["stream_out"], diverging=True)

    # ============================================================
    # WHAT RMSNorm DOES — it rescales each TOKEN (row) to unit RMS.
    # (With untrained weight = all-ones, it ONLY rescales; a trained
    #  weight would also reshape per-dimension.)
    # ============================================================
    print("\n" + "=" * 60)
    print("WHAT RMSNorm DOES — rescales each TOKEN to RMS = 1")
    print("=" * 60)
    norm = layer.input_layernorm

    def per_token_rms(x):
        return [round(r, 2) for r in x.detach().squeeze(0).pow(2).mean(-1).sqrt().tolist()]

    # (a) on the REAL stream-in: already ~unit scale, so the norm barely changes it
    sin = v["stream_in"]
    heatmap("token in  (real stream)", sin, diverging=True)
    heatmap("  -> after RMSNorm", norm(sin), diverging=True)
    print("  per-token RMS before:", per_token_rms(sin))
    print("  per-token RMS after :", per_token_rms(norm(sin)))
    print("  (almost identical — random input is ALREADY ~unit scale; THAT'S why colors match)")

    # (b) on a CRAFTED uneven input: now the norm's equalizing job is visible
    scales = torch.tensor([0.2, 1.0, 3.0, 8.0, 0.5]).view(1, 5, 1)
    uneven = torch.randn(1, 5, config.hidden_size) * scales
    heatmap("token in  (UNEVEN: tok3 huge, tok0 tiny)", uneven, diverging=True)
    heatmap("  -> after RMSNorm (all tokens equalized)", norm(uneven), diverging=True)
    print("  per-token RMS before:", per_token_rms(uneven))
    print("  per-token RMS after :", per_token_rms(norm(uneven)))
    print("  (now you SEE it: wildly different rows -> all the same size)")

    # ============================================================
    # INDUSTRY-STANDARD ATTENTION VISUALIZATION — the attention pattern
    # ("who looks at whom"). Needs TRAINED weights, so this uses GPT-2;
    # the toy layer's attention above is random = noise.
    # ============================================================
    print("\n" + "=" * 60)
    print("ATTENTION PATTERN  (real, trained GPT-2)")
    print("  rows = token doing the looking (QUERY)")
    print("  cols = token being looked at  (KEY)  | bright = strong attention")
    print("  (only the lower triangle: a token can't look at FUTURE tokens)")
    print("=" * 60)

    gtok = AutoTokenizer.from_pretrained("gpt2")
    gpt2 = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager")
    gpt2.eval()
    gids = gtok("The cat sat on the mat", return_tensors="pt").input_ids
    with torch.no_grad():
        gout = gpt2(gids, output_attentions=True)
    gtokens = [gtok.decode(i).strip() or "_" for i in gids[0]]
    print("\n  tokens:", "  ".join(f"{i}={t}" for i, t in enumerate(gtokens)))

    for layer_i, head_i in [(0, 0), (1, 5), (5, 1)]:
        attn_pattern(f"layer {layer_i}, head {head_i}", gout.attentions[layer_i][0, head_i], gtokens)

    print("\n  Look for: a bright DIAGONAL (attend to self/nearby), a bright FIRST")
    print("  COLUMN (everything looks at token 0 = 'attention sink'), or a bright")
    print("  ONE-BELOW-DIAGONAL (each token looks at the PREVIOUS token).")
