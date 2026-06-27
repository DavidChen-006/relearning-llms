import torch
from transformers.modeling_utils import PreTrainedModel

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

    def _init_weights(self, module):
        super()._init_weights(module)
        
