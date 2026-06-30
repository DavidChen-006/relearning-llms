from collections.abc import Callable

from .glm_moe_dsa_pretrained_model import GlmMoeDsaPreTrainedModel
from transformers import GradientCheckpointingLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


class GlmMoeDsaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype

        hidden_states = hidden_states.to(torch.float32)

        variance = hidden_states.pow(2).mean(-1, keepdim=True)

        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return self.weight * hidden_states.to(input_dtype)


class GlmMoeDsaIndexer(nn.Module):


class GlmMoeDsaAttention(nn.Module):
    def __init__(self, config: GlmMoeDsaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.num_heads = config.num_attention_heads          # how many heads
        self.head_dim  = config.hidden_size // self.num_heads # features per head (128 / 4 = 32)
        self.scaling   = self.head_dim ** -0.5                # the 1/√d for the scores

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)  # makes Q
        self.k_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)  # makes K
        self.v_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)  # makes V
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)  # blends heads
    
    forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, #attn_output softmax(QKt/d + M) * V
               torch.Tensor, #attn_weights softmax(QKt/d + M)
               torch.Tensor, 
                | None]:
            
        query_states =  hidden_states @ Wq
        key_states =  hidden_states @ Wk
        value_states = hidden_states @ Wv

        batch, seq = hidden_states.shape[:2]
        query_states = query_states.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        key_states   = key_states.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

        combined_mask = attention_mask

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_inteface( self.config._attn_implementaion, eager_attention_forward)

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            combined_mask,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.transpose(1, 2).reshape(batch, seq, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class GlmMoeDsaMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.down_proj = nn.Linear(config.hidden_size, intermediate_size, bias = False)
        self.up_proj = nn.Linear(intermediate_size, config.hidden_size, bias = False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        self.down_proj(self.act_fn(self.up_proj()))

    # IMPORTANT you ahve to make a demo to watch the values flow through a round of MLP



class GlmMoeDsaTopkRouter(nn.Module):


class GlmMoeDsaNaiveMoe(nn.Module):


class GlmMoeDsaMoE(nn.Module):


class GlmMoeDsaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: GlmMoeDsaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GlmMoeDsaAttention(config, layer_idx)

        self.mlp = GlmMoeDsaMLP()

        self.input_layernorm = GlmMoeDsaRMSNorm()

        self.post_attention_layernorm = GlmMoeDsaRMSNorm()

    forward(
        self
        hidden_states: torch.Tensor
    ) -> :
        residual = hidden_states 
        #norm
        hidden_states = self.input_layernorm(hidden_states)

        #attention
        hidden_states, , = self.self_attn(
            hidden_states,
            ,
            ,
        )

        #add
        hidden_states = hidden_states + residual

        #norm
        hidden_states = self.post_attention_layernorm(hidden_states)

        #mlp
        hidden_states = self.mlp(hidden_states)

        #add
        hidden_states = residual + hidden_states

        return hidden_states


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
        
    def forward(
        self,
        inputs_embeds: torch.FloatTensor | None = None,
    ):
        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        hidden_states = input_embeds

        for decode_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decode_layer(

            )
        
        hidden_states = self.norm(hidden_states)

        return hidden_states
    

class GlmMoeDsaForCausalLM(GlmMoeDsaPreTrainedModel, GenerationMixin):
