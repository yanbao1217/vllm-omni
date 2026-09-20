"""Unit tests for the native Boogu-Image MLLM encoder (vLLM-native online FP8).

Covers: module tree/prefixes, quant_config routing, forward contract
(hidden_states semantics), HF reference parity (text and DeepStack/image
paths), right-padding semantics, and weight mapping/loading integrity.

Parity tolerances follow the plan: hs[0] bitwise equal; per-layer entries
allclose(atol=2e-2, rtol=2e-2) and cosine similarity >= 0.999.
"""

from unittest.mock import MagicMock

import pytest
import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from vllm_omni.diffusion.models.boogu_image.mllm import BooguImageMLLM

pytestmark = [pytest.mark.cpu]

# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="function", autouse=True)
def setup_tp1(monkeypatch, mocker):
    """TP=1 CPU environment (mistral test pattern): identity collectives."""
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", lambda: 0
    )
    mock_tp_group = mocker.MagicMock()
    mock_tp_group.world_size = 1
    mocker.patch("vllm.distributed.parallel_state.get_tp_group", return_value=mock_tp_group)

    def _identity(x, *args, **kwargs):
        return x

    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.tensor_model_parallel_all_reduce", _identity
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.tensor_model_parallel_all_gather", _identity
    )
    mocker.patch("torch.distributed.broadcast")

    from vllm.model_executor.layers.utils import default_unquantized_gemm

    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        lambda: default_unquantized_gemm,
    )

    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        yield


def _tiny_config() -> Qwen3VLConfig:
    cfg = Qwen3VLConfig(
        text_config=dict(
            hidden_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            intermediate_size=128,
            vocab_size=512,
            rms_norm_eps=1e-6,
            rope_theta=500000.0,
            max_position_embeddings=512,
            rope_scaling={
                "mrope_interleaved": True,
                "mrope_section": [8, 8, 8],
                "rope_type": "default",
            },
        ),
        vision_config=dict(
            depth=2,
            hidden_size=32,
            num_heads=4,
            patch_size=14,
            spatial_merge_size=2,
            temporal_patch_size=1,
            intermediate_size=64,
            out_hidden_size=64,
            deepstack_visual_indexes=[0, 1],
            num_position_embeddings=256,
            hidden_act="gelu_pytorch_tanh",
            in_channels=3,
        ),
        image_token_id=10,
        video_token_id=11,
        vision_start_token_id=12,
        vision_end_token_id=13,
        tie_word_embeddings=False,
    )
    cfg.text_config._attn_implementation = "eager"
    return cfg


@pytest.fixture()
def tiny_config():
    torch.manual_seed(0)
    return _tiny_config()


def _image_inputs():
    """Synthetic batch-1 input with one image (grid (1,4,4) -> 4 LLM tokens)."""
    n_img = 4
    pixel_values = torch.randn(1 * 4 * 4, 3 * 1 * 14 * 14)
    image_grid_thw = torch.tensor([[1, 4, 4]])
    input_ids = torch.cat(
        [
            torch.full((1,), 12),
            torch.full((n_img,), 10),
            torch.full((1,), 13),
            torch.randint(20, 500, (6,)),
        ]
    ).view(1, -1)
    mm_token_type_ids = torch.cat(
        [torch.zeros(1), torch.ones(n_img), torch.zeros(7)]
    ).long().view(1, -1)
    attention_mask = torch.ones_like(input_ids)
    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
    )


def _text_inputs(seq: int = 10):
    input_ids = torch.randint(20, 500, (1, seq))
    return dict(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))


# ------------------------------------------------------- construction / tree


class TestConstruction:

    def test_module_tree_mirrors_hf_layout(self, tiny_config):
        mllm = BooguImageMLLM(tiny_config)
        assert hasattr(mllm.visual, "spatial_merge_size")
        assert hasattr(mllm.language_model.embed_tokens, "weight")
        assert len(mllm.language_model.layers) == 3
        assert hasattr(mllm.language_model.norm, "weight")
        names = dict(mllm.language_model.named_parameters())
        assert "layers.0.self_attn.qkv_proj.weight" in names
        assert "layers.0.self_attn.o_proj.weight" in names
        assert "layers.0.mlp.gate_up_proj.weight" in names
        assert "layers.0.mlp.down_proj.weight" in names
        assert "layers.0.self_attn.q_norm.weight" in names
        assert "embed_tokens.weight" in names
        assert "norm.weight" in names

    def test_dtype_property(self, tiny_config):
        mllm = BooguImageMLLM(tiny_config)
        assert mllm.dtype == next(mllm.parameters()).dtype


class TestQuantRouting:
    """MagicMock + UnquantizedLinearMethod keeps the tree CPU-constructible."""

    def test_quant_config_reaches_every_text_linear(self, tiny_config):
        quant_config = MagicMock(name="QuantizationConfig")
        quant_config.get_quant_method.return_value = UnquantizedLinearMethod()
        mllm = BooguImageMLLM(tiny_config, quant_config=quant_config)
        linears = [
            m
            for m in mllm.language_model.modules()
            if hasattr(m, "quant_config") and m.__class__.__name__.endswith("ParallelLinear")
        ]
        assert len(linears) == 4 * 3  # qkv + o + gate_up + down per layer
        for linear in linears:
            assert linear.quant_config is quant_config
        assert quant_config.get_quant_method.call_count == 12
        # Unquantized by construction: vision / embedding / norms untouched.
        assert not hasattr(mllm.visual, "quant_config")
        assert not hasattr(mllm.language_model.embed_tokens, "quant_config")

    def test_linears_get_full_vllm_prefixes(self, tiny_config):
        quant_config = MagicMock(name="QuantizationConfig")
        quant_config.get_quant_method.return_value = UnquantizedLinearMethod()
        BooguImageMLLM(tiny_config, quant_config=quant_config)
        prefixes = [c.kwargs["prefix"] for c in quant_config.get_quant_method.call_args_list]
        assert "mllm.language_model.layers.0.self_attn.qkv_proj" in prefixes
        assert "mllm.language_model.layers.2.mlp.down_proj" in prefixes
        assert all(p.startswith("mllm.language_model.layers.") for p in prefixes)

    def test_fp8_ignored_layers_route_around_fused_prefix(self, tiny_config, mocker):
        """mllm.-prefixed ignored layers -> fused qkv stays unquantized.

        vLLM fused layers require all shards (q+k+v) to share precision —
        partial-shard skip is rejected by ``is_layer_skipped``. Unfused
        linears (o_proj) can be skipped alone.
        """
        quant_config = Fp8Config(
            ignored_layers=[
                "mllm.language_model.layers.0.self_attn.q_proj",
                "mllm.language_model.layers.0.self_attn.k_proj",
                "mllm.language_model.layers.0.self_attn.v_proj",
                "mllm.language_model.layers.1.self_attn.o_proj",
                "transformer.blocks.0.attn.to_q",  # DiT prefix: never matches
            ]
        )
        quant_config.packed_modules_mapping = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            "gate_up_proj": ["gate_proj", "up_proj"],
        }
        # Fp8PerTensorOnlineLinearMethod reads the current VllmConfig's dtype.
        mocker.patch(
            "vllm.model_executor.layers.quantization.online.fp8.get_current_vllm_config",
            return_value=MagicMock(
                model_config=MagicMock(dtype=torch.float32)
            ),
        )
        mllm = BooguImageMLLM(tiny_config, quant_config=quant_config)
        assert isinstance(
            mllm.language_model.layers[0].self_attn.qkv_proj.quant_method,
            UnquantizedLinearMethod,
        )
        assert isinstance(
            mllm.language_model.layers[1].self_attn.o_proj.quant_method,
            UnquantizedLinearMethod,
        )
        assert not isinstance(
            mllm.language_model.layers[1].self_attn.qkv_proj.quant_method,
            UnquantizedLinearMethod,
        )
        assert not isinstance(
            mllm.language_model.layers[2].mlp.gate_up_proj.quant_method,
            UnquantizedLinearMethod,
        )


# ---------------------------------------------------------- forward contract


class TestForwardContract:

    def _run(self, tiny_config, inputs):
        # vLLM parallel linears keep uninitialized buffers until weights load,
        # so always run against the HF reference weights.
        ref = _HFReference(tiny_config)
        mllm = BooguImageMLLM(tiny_config)
        mllm.load_weights(ref.weights())
        with torch.no_grad():
            out = mllm(**inputs, output_hidden_states=True, return_dict=True)
        return mllm, out

    def test_hidden_states_shape_and_last_entry(self, tiny_config):
        _, out = self._run(tiny_config, _text_inputs())
        assert len(out.hidden_states) == 4  # embeddings + 3 layers(末项过 norm)
        assert torch.equal(out.hidden_states[-1], out.last_hidden_state)

    def test_hidden_states_with_image_covers_deepstack(self, tiny_config):
        _, out = self._run(tiny_config, _image_inputs())
        assert len(out.hidden_states) == 4
        shapes = {tuple(h.shape) for h in out.hidden_states}
        assert shapes == {(1, 12, 64)}

    def test_output_hidden_states_false(self, tiny_config):
        mllm = BooguImageMLLM(tiny_config)
        with torch.no_grad():
            out = mllm(**_text_inputs(), output_hidden_states=False)
        assert out.hidden_states is None
        assert out.last_hidden_state.shape == (1, 10, 64)


# ------------------------------------------------------------ HF reference


class _HFReference:
    """Random-init HF tiny model + state_dict adapted for load_weights."""

    def __init__(self, cfg):
        self.model = Qwen3VLForConditionalGeneration(cfg).eval()

    def weights(self):
        # Drop lm_head; keep model.visual.* / model.language_model.* as-is.
        for name, tensor in self.model.state_dict().items():
            if name == "lm_head.weight":
                continue
            yield name, tensor

    def forward(self, **inputs):
        with torch.no_grad():
            return self.model.model(**inputs, output_hidden_states=True, return_dict=True)


def _assert_layer_parity(hf_hs, native_hs):
    assert len(hf_hs) == len(native_hs)
    assert torch.equal(hf_hs[0], native_hs[0]), "embeddings entry must be bitwise equal"
    for i, (ref, got) in enumerate(zip(hf_hs[1:], native_hs[1:])):
        assert torch.allclose(ref.float(), got.float(), atol=2e-2, rtol=2e-2), (
            f"hidden_states[{i + 1}] diverged: max|Δ|="
            f"{(ref.float() - got.float()).abs().max():.3e}"
        )
        cos = torch.nn.functional.cosine_similarity(
            ref.float().flatten(), got.float().flatten(), dim=0
        )
        assert cos >= 0.999, f"hidden_states[{i + 1}] cos={cos:.6f}"


class TestHFReferenceParity:

    def test_text_only(self, tiny_config):
        ref = _HFReference(tiny_config)
        mllm = BooguImageMLLM(tiny_config)
        mllm.load_weights(ref.weights())
        inputs = _text_inputs()
        hf_out = ref.forward(**inputs)
        with torch.no_grad():
            native_out = mllm(**inputs, output_hidden_states=True, return_dict=True)
        _assert_layer_parity(hf_out.hidden_states, native_out.hidden_states)

    def test_image_deepstack(self, tiny_config):
        ref = _HFReference(tiny_config)
        mllm = BooguImageMLLM(tiny_config)
        mllm.load_weights(ref.weights())
        inputs = _image_inputs()
        hf_out = ref.forward(**inputs)
        with torch.no_grad():
            native_out = mllm(**inputs, output_hidden_states=True, return_dict=True)
        _assert_layer_parity(hf_out.hidden_states, native_out.hidden_states)

    def test_right_padding_matches_unpadded(self, tiny_config):
        mllm = BooguImageMLLM(tiny_config)
        mllm.load_weights(_HFReference(tiny_config).weights())
        ids_a = torch.randint(20, 500, (1, 8))
        ids_b = torch.randint(20, 500, (1, 5))
        batched = torch.full((2, 8), 14)  # pad id irrelevant (non-image)
        batched[0, :8] = ids_a
        batched[1, :5] = ids_b
        mask = torch.zeros(2, 8, dtype=torch.long)
        mask[0, :8] = 1
        mask[1, :5] = 1
        with torch.no_grad():
            batched_out = mllm(
                input_ids=batched, attention_mask=mask, output_hidden_states=False
            ).last_hidden_state
            solo_a = mllm(input_ids=ids_a, output_hidden_states=False).last_hidden_state
            solo_b = mllm(input_ids=ids_b, output_hidden_states=False).last_hidden_state
        assert torch.allclose(batched_out[0], solo_a[0], atol=2e-2, rtol=2e-2)
        assert torch.allclose(batched_out[1, :5], solo_b[0], atol=2e-2, rtol=2e-2)


# ------------------------------------------------------------- load weights


class TestLoadWeights:

    def test_fused_concat_order(self, tiny_config):
        ref = _HFReference(tiny_config)
        sd = dict(ref.model.state_dict())
        mllm = BooguImageMLLM(tiny_config)
        mllm.load_weights(ref.weights())
        layer = 1
        got = mllm.language_model.layers[layer].self_attn.qkv_proj.weight
        want = torch.cat(
            [
                sd[f"model.language_model.layers.{layer}.self_attn.q_proj.weight"],
                sd[f"model.language_model.layers.{layer}.self_attn.k_proj.weight"],
                sd[f"model.language_model.layers.{layer}.self_attn.v_proj.weight"],
            ],
            dim=0,
        )
        assert torch.equal(got, want)
        got_mlp = mllm.language_model.layers[layer].mlp.gate_up_proj.weight
        want_mlp = torch.cat(
            [
                sd[f"model.language_model.layers.{layer}.mlp.gate_proj.weight"],
                sd[f"model.language_model.layers.{layer}.mlp.up_proj.weight"],
            ],
            dim=0,
        )
        assert torch.equal(got_mlp, want_mlp)

    def test_vision_weights_map_one_to_one(self, tiny_config):
        ref = _HFReference(tiny_config)
        mllm = BooguImageMLLM(tiny_config)
        mllm.load_weights(ref.weights())
        hf_sd = dict(ref.model.state_dict())
        native_visual = dict(mllm.visual.state_dict())
        for name, ref_t in hf_sd.items():
            if name.startswith("model.visual."):
                stripped = name[len("model.visual.") :]
                assert stripped in native_visual and torch.equal(native_visual[stripped], ref_t)

    def test_missing_fused_shard_raises(self, tiny_config):
        ref = _HFReference(tiny_config)
        weights = [
            (n, t)
            for n, t in ref.weights()
            if ".self_attn.k_proj.weight" not in n
        ]
        mllm = BooguImageMLLM(tiny_config)
        with pytest.raises(RuntimeError, match="missing source shards"):
            mllm.load_weights(weights)

    def test_missing_param_raises_with_full_list(self, tiny_config):
        ref = _HFReference(tiny_config)
        weights = [
            (n, t) for n, t in ref.weights() if ".mlp.down_proj.weight" not in n
        ]
        mllm = BooguImageMLLM(tiny_config)
        with pytest.raises(RuntimeError, match="weights not loaded"):
            mllm.load_weights(weights)
