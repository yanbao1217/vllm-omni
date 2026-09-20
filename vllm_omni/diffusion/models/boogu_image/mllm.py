"""Native Qwen3-VL MLLM encoder for Boogu-Image (vLLM-native online FP8).

Replaces the transformers ``FineGrainedFP8Config`` loading path for the BF16
checkpoints (``Boogu/Boogu-Image-0.1-{Base,Edit}``): the 36-layer text stack is
a native module whose linear layers are vLLM parallel linears governed by
``Fp8Config`` (online quantization is executed by the shared diffusion loader's
``process_weights_after_loading``), mirroring how the DiT transformer consumes
``quant_config``. The vision tower stays the HF module
(``AutoModel.from_config(vision_config)`` — exactly what ``Qwen3VLModel`` does
internally), is never quantized, and is numerically identical to the reference
by construction.

``forward`` replicates transformers 5.17 ``Qwen3VLModel.forward(
output_hidden_states=True)`` semantics, verified empirically against the
installed version (tiny-model hook probe):

    hidden_states = [embeddings, layer_0 .. layer_{N-2} raw outputs,
                     norm(layer_{N-1})]
    hs[-1] == last_hidden_state; per-layer entries are captured BEFORE
    DeepStack injection (injection happens at text layers
    ``0 .. len(deepstack_visual_embeds) - 1``).

Weight mapping (checkpoint ``mllm/`` safetensors names, after the pipeline
loader strips the ``mllm.`` prefix):

    lm_head.weight                         -> dropped
    model.visual.*                         -> visual.* (same HF class, 1:1)
    model.language_model.*                  -> language_model.*
      .self_attn.{q,k,v}_proj.weight        -> .self_attn.qkv_proj.weight (shard q/k/v)
      .mlp.{gate,up}_proj.weight            -> .mlp.gate_up_proj.weight   (shard 0/1)

Reference ports kept numerics-identical to the installed transformers
(5.17.0): rope index (``get_rope_index``, attention-mask aware), interleaved
mrope rotary (``Qwen3VLTextRotaryEmbedding`` imported directly), DeepStack
merge, placeholder scatter.
"""

from dataclasses import dataclass
from itertools import groupby

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


logger = init_logger(__name__)


class BooguImageMLLMRMSNorm(nn.Module):
    """RMSNorm with the reference math (transformers ``Qwen3VLTextRMSNorm``).

    Deliberately unfused: fp32 variance reduction, cast back to the input
    dtype, then the weight multiply — matching the HF encoder the pipeline
    replaced so hidden_states stay numerically comparable. The shared
    ``vllm_omni`` RMSNorm dispatches to a fused CUDA kernel whose precision
    path differs, which shows up directly in the final-norm hidden state.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


@dataclass
class BooguImageMLLMOutput:
    last_hidden_state: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None = None


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _is_quantized_linear(module: nn.Module) -> bool:
    return not isinstance(module.quant_method, UnquantizedLinearMethod)


def _build_attn_bias(
    attention_mask: torch.Tensor | None, dtype: torch.dtype, device: torch.device
) -> torch.Tensor | None:
    """Additive causal+padding bias, (B, 1, S, S); None => pure causal fast path."""
    if attention_mask is None or bool((attention_mask == 1).all()):
        return None
    batch, seq = attention_mask.shape
    causal = torch.ones(seq, seq, dtype=torch.bool, device=device).tril()
    keep = attention_mask[:, None, None, :].to(torch.bool)
    bias = torch.zeros(batch, 1, seq, seq, dtype=dtype, device=device)
    bias.masked_fill_(~(causal[None] & keep), torch.finfo(dtype).min)
    return bias


def _get_vision_position_ids(
    start_position: int,
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Port of transformers 5.17 ``Qwen3VLModel.get_vision_position_ids``."""
    llm_grid_t = grid_thw[0].item()
    llm_grid_h = grid_thw[1].item() // spatial_merge_size
    llm_grid_w = grid_thw[2].item() // spatial_merge_size
    position_temporal = torch.arange(llm_grid_t, device=device)
    position_height = torch.arange(llm_grid_h, device=device) + start_position
    position_width = torch.arange(llm_grid_w, device=device) + start_position
    t_grid, h_grid, w_grid = torch.meshgrid(
        position_temporal, position_height, position_width, indexing="ij"
    )
    vision_position_ids = torch.stack([t_grid, h_grid, w_grid], dim=0).reshape(3, -1)
    vision_position_ids[0] += start_position  # must be after time_interval multiply
    return vision_position_ids


def _get_rope_index(
    hf_config: Qwen3VLConfig,
    input_ids: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Port of transformers 5.17 ``Qwen3VLModel.get_rope_index`` -> (3, B, S).

    Padding-aware: masked positions are zero-filled, matching the reference.
    """
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1
    spatial_merge_size = hf_config.vision_config.spatial_merge_size

    position_ids = torch.zeros(
        3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
    )
    grid_iters = {
        1: iter(image_grid_thw) if image_grid_thw is not None else None,
        2: iter(video_grid_thw) if video_grid_thw is not None else None,
    }

    for batch_idx, current_input_ids in enumerate(input_ids):
        input_token_type = mm_token_type_ids[batch_idx]
        if attention_mask is not None:
            current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
            input_token_type = input_token_type[attention_mask[batch_idx].bool()]

        input_type_group = []
        for key, group in groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
            group = list(group)
            input_type_group.append((key, group[0][0], group[-1][0] + 1))

        current_pos = 0
        llm_pos_ids_list = []
        for modality_type, start_idx, end_idx in input_type_group:
            if modality_type == 0:  # text
                text_len = end_idx - start_idx
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1)
                    + current_pos
                )
                current_pos += text_len
            else:  # image == 1, video == 2
                grid_thw = next(grid_iters[modality_type])
                vision_position_ids = _get_vision_position_ids(
                    current_pos, grid_thw, spatial_merge_size, device=input_ids.device
                )
                llm_pos_ids_list.append(vision_position_ids)
                current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size
        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        if attention_mask is not None:
            position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions.to(
                position_ids.device
            )
        else:
            position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
    return position_ids


class BooguImageMLLMAttention(nn.Module):

    def __init__(
        self,
        text_config,
        *,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.hidden_size = text_config.hidden_size
        total_num_heads = text_config.num_attention_heads
        total_num_kv_heads = text_config.num_key_value_heads
        self.head_dim = getattr(text_config, "head_dim", None) or self.hidden_size // total_num_heads
        self.num_heads = total_num_heads  # TP=1
        self.num_kv_heads = total_num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=total_num_heads,
            total_num_kv_heads=total_num_kv_heads,
            bias=getattr(text_config, "attention_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            total_num_heads * self.head_dim,
            self.hidden_size,
            bias=getattr(text_config, "attention_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = BooguImageMLLMRMSNorm(self.head_dim, eps=text_config.rms_norm_eps)
        self.k_norm = BooguImageMLLMRMSNorm(self.head_dim, eps=text_config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, seq, _ = hidden_states.shape

        # Dual path (MiniMax #5910 pattern): keep the unquantized reference on
        # split GEMMs so bf16 numerics stay aligned with the HF path, where
        # q/k/v are three separate nn.Linear calls.
        if _is_quantized_linear(self.qkv_proj):
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        else:
            weight = self.qkv_proj.weight
            q = F.linear(hidden_states, weight[: self.q_size])
            k = F.linear(hidden_states, weight[self.q_size : self.q_size + self.kv_size])
            v = F.linear(hidden_states, weight[self.q_size + self.kv_size :])

        q = self.q_norm(q.view(batch, seq, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.view(batch, seq, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = v.view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            is_causal=attn_bias is None,
            scale=self.scaling,
            enable_gqa=True,
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch, seq, -1)
        attn_output, _ = self.o_proj(attn_output)
        return attn_output


class BooguImageMLLMMLP(nn.Module):

    def __init__(
        self,
        text_config,
        *,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.intermediate_size = text_config.intermediate_size
        self.act_fn = F.silu

        self.gate_up_proj = MergedColumnParallelLinear(
            text_config.hidden_size,
            [self.intermediate_size, self.intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            text_config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if _is_quantized_linear(self.gate_up_proj):
            gate_up, _ = self.gate_up_proj(hidden_states)
            gate, up = gate_up.split([self.intermediate_size, self.intermediate_size], dim=-1)
        else:
            weight = self.gate_up_proj.weight
            gate = F.linear(hidden_states, weight[: self.intermediate_size])
            up = F.linear(hidden_states, weight[self.intermediate_size :])
        down, _ = self.down_proj(self.act_fn(gate) * up)
        return down


class BooguImageMLLMDecoderLayer(nn.Module):

    def __init__(
        self,
        text_config,
        *,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.self_attn = BooguImageMLLMAttention(
            text_config, quant_config=quant_config, prefix=f"{prefix}.self_attn"
        )
        self.mlp = BooguImageMLLMMLP(text_config, quant_config=quant_config, prefix=f"{prefix}.mlp")
        self.input_layernorm = BooguImageMLLMRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.post_attention_layernorm = BooguImageMLLMRMSNorm(
            text_config.hidden_size, eps=text_config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin, attn_bias)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class BooguImageMLLMTextModel(nn.Module):
    """36-layer text stack with vLLM-quantized linears and HF hidden_states semantics."""

    def __init__(
        self,
        text_config,
        *,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.config = text_config
        self.embed_tokens = nn.Embedding(text_config.vocab_size, text_config.hidden_size)
        self.layers = nn.ModuleList(
            BooguImageMLLMDecoderLayer(
                text_config, quant_config=quant_config, prefix=f"{prefix}.layers.{i}"
            )
            for i in range(text_config.num_hidden_layers)
        )
        self.norm = BooguImageMLLMRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(text_config)

    @staticmethod
    def _deepstack_process(
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Port of transformers 5.17 ``Qwen3VLTextModel._deepstack_process``."""
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        local_this = hidden_states[visual_pos_masks, :] + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,  # (3, B, S)
        attn_bias: torch.Tensor | None,
        *,
        output_hidden_states: bool = False,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        hidden_states = inputs_embeds
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        num_layers = len(self.layers)

        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, cos, sin, attn_bias)
            # HF semantics (verified empirically on transformers 5.17): the
            # last layer's raw output is NOT captured — its tuple slot holds
            # the final-normed value appended after the loop. Entries for
            # layers 0..N-2 are captured BEFORE DeepStack injection.
            if output_hidden_states and layer_idx < num_layers - 1:
                all_hidden_states += (hidden_states,)
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states, visual_pos_masks, deepstack_visual_embeds[layer_idx]
                )

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        return hidden_states, all_hidden_states


class BooguImageMLLM(nn.Module):
    """Hybrid Qwen3-VL encoder: HF vision tower (never quantized) + native text stack."""

    def __init__(
        self,
        hf_config: Qwen3VLConfig,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "mllm",
    ) -> None:
        super().__init__()
        self.config = hf_config
        # Same construction the HF reference uses internally
        # (Qwen3VLModel.__init__: self.visual = AutoModel.from_config(...)).
        self.visual = AutoModel.from_config(hf_config.vision_config)
        self.language_model = BooguImageMLLMTextModel(
            hf_config.text_config,
            quant_config=quant_config,
            prefix=f"{prefix}.language_model",
        )

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def _vision_forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Merged image embeds (num_tokens, hidden) + deepstack feature lists."""
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_output = self.visual(pixel_values, grid_thw=grid_thw, return_dict=True)
        image_embeds = vision_output.pooler_output
        split_sizes = (grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.cat(torch.split(image_embeds, split_sizes), dim=0)
        return image_embeds, vision_output.deepstack_features

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ) -> BooguImageMLLMOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)

        image_mask = None
        deepstack_visual_embeds = None
        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self._vision_forward(pixel_values, image_grid_thw)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = input_ids == self.config.image_token_id
            n_image_tokens = image_mask.sum()
            if n_image_tokens * inputs_embeds.shape[-1] != image_embeds.numel():
                raise ValueError(
                    f"Image features and image tokens do not match, tokens: {n_image_tokens}, "
                    f"features: {image_embeds.shape[0]}"
                )
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.unsqueeze(-1).to(inputs_embeds.device), image_embeds
            )
            deepstack_visual_embeds = deepstack_image_embeds
        if pixel_values_videos is not None:
            raise NotImplementedError("Boogu-Image MLLM does not receive video inputs")

        if position_ids is None:
            if image_grid_thw is not None or video_grid_thw is not None:
                if mm_token_type_ids is None:
                    raise ValueError(
                        "Multimodal data was passed but `mm_token_type_ids` is missing."
                    )
                position_ids = _get_rope_index(
                    self.config,
                    input_ids,
                    mm_token_type_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                )
            else:
                batch, seq = inputs_embeds.shape[:2]
                position_ids = (
                    torch.arange(seq, device=inputs_embeds.device)
                    .view(1, 1, -1)
                    .expand(3, batch, seq)
                )

        visual_pos_masks = image_mask if image_mask is not None else None
        attn_bias = _build_attn_bias(attention_mask, inputs_embeds.dtype, inputs_embeds.device)

        hidden_states, all_hidden_states = self.language_model(
            inputs_embeds,
            position_ids,
            attn_bias,
            output_hidden_states=output_hidden_states,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        return BooguImageMLLMOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states
        )

    def _map_weight_name(self, name: str) -> tuple[str, str | int | None] | None:
        if name == "lm_head.weight":
            return None
        if name.startswith("model.visual."):
            return ("visual." + name[len("model.visual.") :], None)
        if name.startswith("model.language_model."):
            rest = name[len("model.language_model.") :]
            for suffix, target, shard in (
                (".self_attn.q_proj.weight", ".self_attn.qkv_proj.weight", "q"),
                (".self_attn.k_proj.weight", ".self_attn.qkv_proj.weight", "k"),
                (".self_attn.v_proj.weight", ".self_attn.qkv_proj.weight", "v"),
                (".mlp.gate_proj.weight", ".mlp.gate_up_proj.weight", 0),
                (".mlp.up_proj.weight", ".mlp.gate_up_proj.weight", 1),
            ):
                if rest.endswith(suffix):
                    return ("language_model." + rest[: -len(suffix)] + target, shard)
            return ("language_model." + rest, None)
        return None

    def load_weights(self, weights) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        # Fused completeness: qkv needs {q,k,v}, gate_up needs {0,1} per layer.
        expected_fused: dict[str, set] = {}
        for name in params:
            if name.endswith(".self_attn.qkv_proj.weight"):
                expected_fused[name] = {"q", "k", "v"}
            elif name.endswith(".mlp.gate_up_proj.weight"):
                expected_fused[name] = {0, 1}
        loaded_fused: dict[str, set] = {name: set() for name in expected_fused}

        for name, tensor in weights:
            mapped = self._map_weight_name(name)
            if mapped is None:
                continue
            param_name, shard_id = mapped
            param = params.get(param_name)
            if param is None:
                logger.warning("Boogu MLLM weight %s has no target parameter", name)
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            if shard_id is None:
                weight_loader(param, tensor)
            else:
                weight_loader(param, tensor, shard_id)
            loaded.add(param_name)
            if shard_id is not None and param_name in loaded_fused:
                loaded_fused[param_name].add(shard_id)

        for param_name, expected_shards in expected_fused.items():
            missing_shards = expected_shards - loaded_fused[param_name]
            if missing_shards:
                raise RuntimeError(
                    f"Boogu MLLM fused weight {param_name} missing source shards {missing_shards}"
                )
        missing = sorted(set(params) - loaded)
        if missing:
            # Listed in full: this aborts startup, so the message is the only diagnostic.
            raise RuntimeError(f"Boogu MLLM weights not loaded: {len(missing)} params: {missing}")
        return loaded
