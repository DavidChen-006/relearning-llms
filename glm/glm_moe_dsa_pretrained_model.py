"""GlmMoeDsaPreTrainedModel — copied verbatim from the real modeling_glm_moe_dsa.py
(lines 662-691). This is the lightweight base class every GLM-MoE-DSA model inherits:
it holds metadata flags + the weight-initialization rule. It does NOT do a forward pass.

DEPENDENCIES this class references (must exist in your module for it to actually run):
  - GlmMoeDsaConfig        (your config class)
  - GlmMoeDsaDecoderLayer  )
  - GlmMoeDsaAttention     )  sibling classes from the modeling file
  - GlmMoeDsaTopkRouter    )
  - GlmMoeDsaExperts       )
"""
import torch
import torch.nn.init as init
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import auto_docstring

# from .configuration_glm_moe_dsa import GlmMoeDsaConfig   # <- your config
# from .modeling_glm_moe_dsa import (                      # <- your sibling classes
#     GlmMoeDsaDecoderLayer, GlmMoeDsaAttention, GlmMoeDsaTopkRouter, GlmMoeDsaExperts,
# )


@auto_docstring
class GlmMoeDsaPreTrainedModel(PreTrainedModel):
    config: GlmMoeDsaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GlmMoeDsaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = False  # flash-mla kernels need a bit more work in the way we enable them!
    _supports_sdpa = True
    _supports_flex_attn = False

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": GlmMoeDsaDecoderLayer,
        "attentions": GlmMoeDsaAttention,
    }
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.78.*"]
    _keep_in_fp32_modules = ["indexer.weights_proj"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, GlmMoeDsaTopkRouter):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, GlmMoeDsaExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
