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
