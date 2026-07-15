"""Day-1 config — skeleton shape only, before MoE / MLA / DSA exist."""

from transformers.configuration_utils import PreTrainedConfig


class GlmMoeDsaConfig(PreTrainedConfig):
    # Identity: which architecture family this config belongs to.
    model_type = "glm_moe_dsa"

    # Toy scale — same fields you'd pick before writing any modeling code.
    # Enough to size: embed_tokens → [decoder layers] → norm → lm_head
    vocab_size: int = 96
    hidden_size: int = 128
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    intermediate_size: int = 256
    max_position_embeddings: int = 256
    hidden_act: str = "silu"
    # Per-layer MLP kind: "dense" (GlmMoeDsaMLP) or "sparse" (GlmMoeDsaMoE).
    # Mirrors glm/reference_config.py — default: first 3 dense, rest sparse.
    mlp_layer_types: list[str] | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] * min(3, self.num_hidden_layers) + ["sparse"] * (
                self.num_hidden_layers - 3
            )
