from .glm_moe_dsa_pretrained_model import GlmMoeDsaPreTrainedModel
from transformers import GradientCheckpointingLayer

class GlmMoeDsaRMSNorm(nn.Module):


class GlmMoeDsaIndexer(nn.Module):


class GlmMoeDsaAttention(nn.Module):


class GlmMoeDsaMLP(nn.Module):


class GlmMoeDsaTopkRouter(nn.Module):


class GlmMoeDsaNaiveMoe(nn.Module):


class GlmMoeDsaMoE(nn.Module):


class GlmMoeDsaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: GlmMoeDsaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GlmMoeDsaAttention()

        self.mlp = GlmMoeDsaMLP()

        self.input_layernorm = GlmMoeDsaRMSNorm()

        self.post_attention_layernorm = GlmMoeDsaRMSNorm()



class GlmMoeDsaRotaryEmbedding(nn.Module):


class GlmMoeDsaModel(GlmMoeDsaPreTrainedModel):
    def __init__(self, config: GlmMoeDsaConfig):
        super().__init__(config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GlmMoeDsaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GlmMoeDsaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = GlmMoeDsaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False



class GlmMoeDsaForCausalLM(GlmMoeDsaPreTrainedModel, GenerationMixin):
