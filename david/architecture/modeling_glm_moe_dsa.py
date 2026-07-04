from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, GlmMoeDsaConfig, GradientCheckpointingLayer
from transformers.activations import ACT2FN
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel


# the plain "eager" attention: softmax(QKt * scaling + mask) @ V  (fallback for the interface)
def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, value)   # (batch, heads, seq, head_dim)
    return attn_output, attn_weights

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
    ) -> torch.Tensor:

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    x_rotated = (x * cos) + (rotate_half(x) * sin)

    return x_rotated


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


class GlmMoeDsaIndexer(nn.Module):   # DSA add-on — deferred
    pass


class GlmMoeDsaAttention(nn.Module):
    def __init__(self, config: GlmMoeDsaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.num_heads = config.num_attention_heads          # how many heads
        self.head_dim = config.hidden_size // self.num_heads  # features per head (128 / 4 = 32)
        self.scaling = self.head_dim ** -0.5                 # the 1/√d for the scores

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)  # makes Q
        self.k_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)  # makes K
        self.v_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)  # makes V
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)  # blends heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:   # (attn_output, attn_weights)

        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states)   # FIX: was `hidden_states @ Wq`
        key_states = self.k_proj(hidden_states)


        value_states = self.v_proj(hidden_states)

        batch, seq = hidden_states.shape[:2]
        query_states = query_states.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

        query_states = apply_rotary_pos_emb(query_states, cos, sin)
        key_states = apply_rotary_pos_emb(key_states, cos, sin)

        combined_mask = attention_mask

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(   # FIX: typo get_inteface
            self.config._attn_implementation, eager_attention_forward           # FIX: typo _attn_implementaion
        )

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
        intermediate_size = intermediate_size or config.intermediate_size
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)     # FIX: expand hidden->inter
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)   # FIX: shrink inter->hidden
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))   # FIX: up_proj(x) + return


class GlmMoeDsaTopkRouter(nn.Module):   # MoE add-on — deferred
    pass


class GlmMoeDsaNaiveMoe(nn.Module):   # MoE add-on — deferred
    pass


class GlmMoeDsaMoE(nn.Module):   # MoE add-on — deferred
    pass


class GlmMoeDsaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: GlmMoeDsaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GlmMoeDsaAttention(config, layer_idx)
        self.mlp = GlmMoeDsaMLP(config, config.intermediate_size)                          # FIX: pass config
        self.input_layernorm = GlmMoeDsaRMSNorm(config.hidden_size, config.rms_norm_eps)   # FIX: pass args
        self.post_attention_layernorm = GlmMoeDsaRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, 
                hidden_states: torch.Tensor, 
                attention_mask: torch.Tensor | None = None, 
                position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> torch.Tensor:

        residual = hidden_states
        # norm
        hidden_states = self.input_layernorm(hidden_states)
        # attention
        hidden_states, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )   # FIX: filled args + unpack
        # add
        hidden_states = residual + hidden_states
        # norm
        residual = hidden_states                                          # FIX: save the post-attn residual
        hidden_states = self.post_attention_layernorm(hidden_states)
        # mlp
        hidden_states = self.mlp(hidden_states)
        # add
        hidden_states = residual + hidden_states
        return hidden_states


class GlmMoeDsaRotaryEmbedding(nn.Module):   # the angle factory: positions in -> (cos, sin) out
    def __init__(self, config: GlmMoeDsaConfig, device=None):
        super().__init__()
        dim = config.hidden_size // config.num_attention_heads      # head_dim: RoPE acts per head
        base = config.rope_parameters["rope_theta"]                 # the 10000 in the speed formula
        # spin speeds, one per pair: theta_i = base^(-2i/dim)  (fast hands first, slow last)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)   # fixed math, not a weight

    @torch.no_grad()
    def forward(self, x, position_ids):
        # angle grid: every position x every speed -> (batch, seq, dim/2)
        angles = position_ids[:, :, None].float() * self.inv_freq[None, None, :]
        emb = torch.cat((angles, angles), dim=-1)    # duplicate so both partner slots share an angle
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class GlmMoeDsaPreTrainedModel(PreTrainedModel):   # merged in from the separate file (fixes circular import)
    config: GlmMoeDsaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GlmMoeDsaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _can_record_outputs = {
        "hidden_states": GlmMoeDsaDecoderLayer,
        "attentions": GlmMoeDsaAttention,
    }

    def _init_weights(self, module):
        super()._init_weights(module)


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

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,       # FIX: added input_ids
        inputs_embeds: torch.FloatTensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds                    # FIX: was input_embeds (typo)
        # positions are just 0..seq-1, one row shared by the whole batch
        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        # causal mask: -inf above the diagonal so a token can't see the future
        seq_len = hidden_states.shape[1]
        causal_mask = torch.full((seq_len, seq_len), float("-inf"), device=hidden_states.device).triu(1)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(hidden_states, 
                causal_mask,
                position_embeddings=position_embeddings)   # FIX: filled args

        hidden_states = self.norm(hidden_states)
        return hidden_states


class GlmMoeDsaForCausalLM(GlmMoeDsaPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.model = GlmMoeDsaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:   # FIX: added input_ids
        hidden_states = self.model(input_ids)          # FIX: pass input_ids; model returns a tensor
        logits = self.lm_head(hidden_states)           # produce logits
        return logits                                  # FIX: added return

    def generate(self, input_ids, max_new_tokens=20, eos_token_id=None):
        """The autoregressive loop (HF's version lives in GenerationMixin.generate):
        forward -> pick next token -> append -> stop on EOS or the max ceiling."""
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self(input_ids)                       # (batch, seq, vocab)
            next_token = logits[0, -1].argmax()                # greedy decoding
            input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break                                          # model says "I'm done"
        return input_ids
