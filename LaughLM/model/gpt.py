"""
LaughLM/model/gpt.py

Top-level GPT model — nn.scan for training, for-loop for inference.

Key design:
- Training (kv_caches=None): uses nn.scan when scan_layers=True for O(1) compile
- Inference (kv_caches != None): uses a for-loop with per-layer params extracted
  from the scan variable tree

FIX (audit 2025): Previous code created UNINITIALIZED TransformerBlock instances
during inference when scan_layers=True, producing GARBAGE output.

The fix: when scan_layers=True, inference extracts per-layer params from the
scanned param tree via _unstack_scan_params() and uses .apply() to run each
block statelessly. When scan_layers=False, blocks run normally via self.blocks.

A single "reference block" is created for type/structure only during scan mode —
it's used via .apply() with per-layer params, never via __call__ with its own params.

FIX (2026-05-06): _unstack_scan_params used isinstance(tree, jnp.ndarray) to detect
leaf arrays to split. After flax.serialization.from_bytes(), params become plain
numpy.ndarray instances (NOT jnp.ndarray), causing the check to fail and the
scanned params to be returned un-split. The reference block then received
stacked params with shape (num_layers, d_model) instead of per-layer (d_model,),
triggering flax.errors.ScopeParamShapeError. Fixed by using duck-typing
(hasattr ndim + shape) instead of isinstance.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple, List

from LaughLM.config.schema import LaughLMConfig
from LaughLM.model.transformer_block import TransformerBlock, build_block, get_remat_policy
from LaughLM.model.layers.normalization import build_normalization
from LaughLM.model.layers.positional import (
    build_positional_encoding,
    build_rope_tables,
)
from LaughLM.model.layers.attention import KVCache
from LaughLM.utils.dtype import resolve_compute_dtype


def _build_scanned_block(config: LaughLMConfig):
    """
    Build a scanned transformer stack using nn.scan.
    Params stacked [num_layers, ...]. Single XLA trace reused N times.
    """
    remat_cfg = config.spmd.remat

    BlockClass = TransformerBlock

    if remat_cfg.policy != "everything_saveable":
        policy = get_remat_policy(remat_cfg.policy)
        BlockClass = nn.remat(
            BlockClass,
            policy=policy,
            prevent_cse=remat_cfg.prevent_cse,
        )

    ScanBlock = nn.scan(
        BlockClass,
        variable_axes={"params": 0},
        split_rngs={"params": True},
        in_axes=(nn.broadcast, nn.broadcast, nn.broadcast),
        length=config.model.num_layers,
    )

    return ScanBlock(config=config)


def _unstack_scan_params(params, num_layers):
    """
    Convert scanned param tree → list of per-layer param dicts.

    nn.scan with variable_axes={"params": 0} stacks each scanned variable
    along axis 0. This function recursively walks the param dict and for
    each leaf ndarray with shape[0] == num_layers, splits it into
    num_layers slices. Other leaves (non-scanned params like embedding tables)
    are kept unchanged.

    Returns: list of num_layers param dicts, each structured like a
    single TransformerBlock's params.
    """

    def _is_array(tree):
        """Check if tree is an ndarray-like (JAX or numpy).

        After flax.serialization.from_bytes(), params become plain
        numpy.ndarray instances, NOT jnp.ndarray. Duck-typing by
        ndim + shape handles both cases.
        """
        return hasattr(tree, 'ndim') and hasattr(tree, 'shape')

    def _split(tree):
        """Recursively split a param tree. Returns either the original
        (non-scanned) or a list of per-layer dicts (scanned)."""
        if isinstance(tree, dict):
            keys = sorted(tree.keys())
            # Recursively split each child
            split_children = {k: _split(tree[k]) for k in keys}

            # Determine if any child was split (returned a list)
            any_split = any(isinstance(v, list) for v in split_children.values())

            if any_split:
                # Merge per-layer dicts across all children
                result = []
                for i in range(num_layers):
                    layer_dict = {}
                    for k in keys:
                        child = split_children[k]
                        if isinstance(child, list):
                            layer_dict[k] = child[i]
                        else:
                            # Non-scanned: same across all layers
                            layer_dict[k] = child
                    result.append(layer_dict)
                return result
            else:
                return tree
        elif _is_array(tree):
            if tree.ndim > 0 and tree.shape[0] == num_layers:
                return [tree[i] for i in range(num_layers)]
            else:
                return tree
        else:
            return tree

    # The scan_block subtree contains stacked per-layer params
    # Structure: scan_block → {Dense_0: {kernel: [L, ...], bias: [L, ...]}, ...}
    result = _split(params)

    if isinstance(result, list):
        return result
    elif isinstance(result, dict):
        # No scanned params found — this means all params are non-scanned
        # which shouldn't happen for scan_block. Return as replicated.
        return [result] * num_layers
    else:
        raise ValueError(f"Unexpected result from _split: {type(result)}")


class GPTModel(nn.Module):
    config: LaughLMConfig

    def setup(self):
        cfg = self.config
        d_model = cfg.model.d_model
        vocab_size = cfg.model.vocab_size
        num_layers = cfg.model.num_layers
        pos_type = cfg.architecture.positional

        self._compute_dtype = resolve_compute_dtype(cfg)
        self._use_scan = cfg.spmd.remat.scan_layers
        self._num_layers = num_layers

        # ── Token embedding ───────────────────────────────────
        self.token_embedding = nn.Embed(
            num_embeddings=vocab_size,
            features=d_model,
            embedding_init=nn.initializers.normal(
                stddev=cfg.initialization.embedding_std
            ),
        )

        # ── Positional encoding (additive only) ──────────────
        self.positional = build_positional_encoding(cfg)

        # ── RoPE tables ───────────────────────────────────────
        self._use_rope = pos_type in ("rope", "rope_scaled")

        if self._use_rope:
            head_dim = d_model // cfg.model.num_heads
            scale_factor = 4.0 if pos_type == "rope_scaled" else None
            self._rope_sin, self._rope_cos = build_rope_tables(
                head_dim=head_dim,
                max_seq_len=cfg.model.max_seq_len,
                scale_factor=scale_factor,
            )
        else:
            self._rope_sin = None
            self._rope_cos = None

        # ── Transformer blocks ────────────────────────────────
        if self._use_scan:
            # Scan mode: use nn.scan for training (O(1) compile)
            # Also create a reference block for inference .apply() calls
            self.scan_block = _build_scanned_block(cfg)
            self._ref_block = TransformerBlock(config=cfg)
        else:
            # Non-scan mode: explicit blocks for both training and inference
            self.blocks = [build_block(cfg) for _ in range(num_layers)]

        # ── Final norm ────────────────────────────────────────
        self.final_norm = build_normalization(cfg)

        # ── LM head ──────────────────────────────────────────
        if not cfg.architecture.weight_tying:
            self.lm_head = nn.Dense(
                vocab_size,
                use_bias=cfg.architecture.bias,
                kernel_init=nn.initializers.normal(stddev=cfg.initialization.std),
            )

    def __call__(
        self,
        input_ids: jnp.ndarray,
        doc_ids: Optional[jnp.ndarray] = None,
        kv_caches: Optional[List[KVCache]] = None,
    ) -> Tuple[jnp.ndarray, Optional[List[KVCache]]]:

        assert input_ids.ndim == 2, f"Expected (B, T), got {input_ids.shape}"
        B, T = input_ids.shape

        # ── Token embedding ───────────────────────────────────
        x = self.token_embedding(input_ids)
        x = x.astype(self._compute_dtype)

        # ── Positional encoding ───────────────────────────────
        if self.positional is not None:
            positions = jnp.arange(T)[None, :]
            pos_emb = self.positional(positions)
            x = x + pos_emb.astype(self._compute_dtype)

        # ── RoPE tables ───────────────────────────────────────
        rope_tables = None
        if self._use_rope:
            rope_tables = (self._rope_sin[:T], self._rope_cos[:T])

        # ── Transformer stack ─────────────────────────────────
        if kv_caches is not None:
            # ── Inference: for-loop with per-layer KV cache ──
            new_caches = []

            if self._use_scan:
                # Extract per-layer params from the scanned parameter tree.
                # self.variables['params'] contains:
                #   {token_embedding: {...}, scan_block: {Dense_0: {kernel: [L, ...]}, ...}, ...}
                # We need just the scan_block subtree, unstacked per layer.
                all_params = self.variables.get("params", {})
                scan_params = all_params.get("scan_block", all_params)
                layer_params_list = _unstack_scan_params(scan_params, self._num_layers)

                for i in range(self._num_layers):
                    # Use .apply() with per-layer params — stateless, no init needed
                    block_vars = {"params": layer_params_list[i]}
                    x, new_cache = self._ref_block.apply(
                        block_vars,
                        x,
                        rope_tables=rope_tables,
                        doc_ids=doc_ids,
                        kv_cache=kv_caches[i],
                    )
                    new_caches.append(new_cache)
            else:
                for i, block in enumerate(self.blocks):
                    x, new_cache = block(
                        x,
                        rope_tables=rope_tables,
                        doc_ids=doc_ids,
                        kv_cache=kv_caches[i],
                    )
                    new_caches.append(new_cache)

        elif self._use_scan:
            # ── Training: nn.scan (O(1) compile, optimal) ──
            x, _ = self.scan_block(x, rope_tables, doc_ids, None)
            new_caches = None

        else:
            # ── Fallback: for-loop (no scan) ──
            for block in self.blocks:
                x, _ = block(x, rope_tables=rope_tables, doc_ids=doc_ids, kv_cache=None)
            new_caches = None

        # ── Final norm + logits ───────────────────────────────
        x = self.final_norm(x)
        x = x.astype(jnp.float32)

        if self.config.architecture.weight_tying:
            embedding_table = self.token_embedding.embedding
            logits = jnp.einsum("btd,vd->btv", x, embedding_table)
        else:
            logits = self.lm_head(x)

        return logits, new_caches