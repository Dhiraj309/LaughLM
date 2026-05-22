"""
LaughLM/model/llama/remat.py

Selective activation checkpointing utilities.
"""

import jax

from flax import linen as nn


# ============================================================
# Policy dispatch
# ============================================================

def get_remat_policy(
    policy_name: str,
):

    policies = jax.checkpoint_policies

    if policy_name == "nothing_saveable":
        return policies.nothing_saveable

    if policy_name == "dots_saveable":
        return policies.dots_saveable

    if (
        policy_name
        == "dots_with_no_batch_dims_saveable"
    ):
        return (
            policies
            .dots_with_no_batch_dims_saveable
        )

    if policy_name == "everything_saveable":
        return policies.everything_saveable

    raise ValueError(
        f"Unknown remat policy: {policy_name}"
    )


# ============================================================
# New API
# ============================================================

def maybe_remat(
    module_cls,
    *,
    enabled: bool,
    policy: str | None,
    prevent_cse: bool = False,
):
    """
    Conditionally wrap module with remat.
    """

    if not enabled:

        return module_cls

    if policy is None:

        raise ValueError(
            "remat enabled but "
            "remat_policy is None"
        )

    remat_policy = get_remat_policy(
        policy
    )

    return nn.remat(
        module_cls,
        policy=remat_policy,
        prevent_cse=prevent_cse,
    )


# ============================================================
# Backward-compatible API
# ============================================================

def remat_module(
    module_cls,
    *,
    policy: str,
    prevent_cse: bool = False,
):
    """
    Backward-compatible wrapper.

    Older model.py versions still call:
        remat_module(...)

    Internally forwarded into maybe_remat.
    """

    return maybe_remat(
        module_cls,
        enabled=True,
        policy=policy,
        prevent_cse=prevent_cse,
    )
