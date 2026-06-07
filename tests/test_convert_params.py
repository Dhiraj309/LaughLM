from types import SimpleNamespace

import numpy as np

from LaughLM.export.convert_params import convert_params_to_hf


def _cfg(*, tie_word_embeddings=True):
    return SimpleNamespace(
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=2,
        num_hidden_layers=1,
        tie_word_embeddings=tie_word_embeddings,
    )


def _dense(in_features, out_features, offset=0):
    kernel = np.arange(
        offset,
        offset + in_features * out_features,
        dtype=np.float32,
    ).reshape(in_features, out_features)

    return {"kernel": kernel}


def _norm(size, offset=0):
    return {
        "weight": np.arange(
            offset,
            offset + size,
            dtype=np.float32,
        )
    }


def _base_params(*, fused_qkv=True, tie_word_embeddings=True):
    hidden = 4
    q_dim = 4
    kv_dim = 4

    if fused_qkv:
        self_attn = {
            "qkv_proj": _dense(hidden, q_dim + 2 * kv_dim, 0),
            "o_proj": _dense(hidden, hidden, 100),
        }
    else:
        self_attn = {
            "q_proj": _dense(hidden, q_dim, 0),
            "k_proj": _dense(hidden, kv_dim, 100),
            "v_proj": _dense(hidden, kv_dim, 200),
            "o_proj": _dense(hidden, hidden, 300),
        }

    layer = {
        "self_attn": self_attn,
        "mlp": {
            "gate_proj": _dense(hidden, 8, 400),
            "up_proj": _dense(hidden, 8, 500),
            "down_proj": _dense(8, hidden, 600),
        },
        "input_layernorm": _norm(hidden, 700),
        "post_attention_layernorm": _norm(hidden, 800),
    }

    params = {
        "model": {
            "embed_tokens": {
                "embedding": np.arange(
                    10 * hidden,
                    dtype=np.float32,
                ).reshape(10, hidden)
            },
            "layers_0": layer,
            "norm": _norm(hidden, 900),
        }
    }

    if not tie_word_embeddings:
        params["lm_head"] = _dense(hidden, 10, 1000)

    return params


def test_convert_fused_qkv_split_order_and_transpose():
    cfg = _cfg(tie_word_embeddings=True)
    params = _base_params(
        fused_qkv=True,
        tie_word_embeddings=True,
    )

    tensors = convert_params_to_hf(
        params=params,
        config=cfg,
    )

    qkv_kernel = params["model"]["layers_0"]["self_attn"]["qkv_proj"]["kernel"]

    expected_q = qkv_kernel[:, 0:4].T
    expected_k = qkv_kernel[:, 4:8].T
    expected_v = qkv_kernel[:, 8:12].T

    np.testing.assert_array_equal(
        tensors["model.layers.0.self_attn.q_proj.weight"],
        expected_q,
    )

    np.testing.assert_array_equal(
        tensors["model.layers.0.self_attn.k_proj.weight"],
        expected_k,
    )

    np.testing.assert_array_equal(
        tensors["model.layers.0.self_attn.v_proj.weight"],
        expected_v,
    )


def test_convert_unfused_qkv_transpose():
    cfg = _cfg(tie_word_embeddings=True)
    params = _base_params(
        fused_qkv=False,
        tie_word_embeddings=True,
    )

    tensors = convert_params_to_hf(
        params=params,
        config=cfg,
    )

    attn = params["model"]["layers_0"]["self_attn"]

    np.testing.assert_array_equal(
        tensors["model.layers.0.self_attn.q_proj.weight"],
        attn["q_proj"]["kernel"].T,
    )

    np.testing.assert_array_equal(
        tensors["model.layers.0.self_attn.k_proj.weight"],
        attn["k_proj"]["kernel"].T,
    )

    np.testing.assert_array_equal(
        tensors["model.layers.0.self_attn.v_proj.weight"],
        attn["v_proj"]["kernel"].T,
    )


def test_convert_mlp_gate_up_down_not_swapped():
    cfg = _cfg(tie_word_embeddings=True)
    params = _base_params(
        fused_qkv=True,
        tie_word_embeddings=True,
    )

    tensors = convert_params_to_hf(
        params=params,
        config=cfg,
    )

    mlp = params["model"]["layers_0"]["mlp"]

    np.testing.assert_array_equal(
        tensors["model.layers.0.mlp.gate_proj.weight"],
        mlp["gate_proj"]["kernel"].T,
    )

    np.testing.assert_array_equal(
        tensors["model.layers.0.mlp.up_proj.weight"],
        mlp["up_proj"]["kernel"].T,
    )

    np.testing.assert_array_equal(
        tensors["model.layers.0.mlp.down_proj.weight"],
        mlp["down_proj"]["kernel"].T,
    )


def test_convert_tied_embeddings_omits_lm_head():
    cfg = _cfg(tie_word_embeddings=True)
    params = _base_params(
        fused_qkv=True,
        tie_word_embeddings=True,
    )

    tensors = convert_params_to_hf(
        params=params,
        config=cfg,
    )

    assert "model.embed_tokens.weight" in tensors
    assert "lm_head.weight" not in tensors


def test_convert_untied_embeddings_exports_lm_head():
    cfg = _cfg(tie_word_embeddings=False)
    params = _base_params(
        fused_qkv=True,
        tie_word_embeddings=False,
    )

    tensors = convert_params_to_hf(
        params=params,
        config=cfg,
    )

    assert "model.embed_tokens.weight" in tensors
    assert "lm_head.weight" in tensors

    np.testing.assert_array_equal(
        tensors["lm_head.weight"],
        params["lm_head"]["kernel"].T,
    )


def test_convert_norm_weights_are_not_transposed():
    cfg = _cfg(tie_word_embeddings=True)
    params = _base_params(
        fused_qkv=True,
        tie_word_embeddings=True,
    )

    tensors = convert_params_to_hf(
        params=params,
        config=cfg,
    )

    np.testing.assert_array_equal(
        tensors["model.layers.0.input_layernorm.weight"],
        params["model"]["layers_0"]["input_layernorm"]["weight"],
    )

    np.testing.assert_array_equal(
        tensors["model.layers.0.post_attention_layernorm.weight"],
        params["model"]["layers_0"]["post_attention_layernorm"]["weight"],
    )

    np.testing.assert_array_equal(
        tensors["model.norm.weight"],
        params["model"]["norm"]["weight"],
    )
