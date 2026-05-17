"""
LaughLM/model/llama/remat.py

Activation checkpointing utilities for LaughLM.

Design goals
-------------
- MaxText-style rematerialization
- policy-driven checkpointing
- deterministic wrapping
- scan-compatible structure
- TPU memory reduction

References
----------
- MaxText
- T5X
- Pax
- JAX checkpoint_policies
"""

import jax

from flax import linen as nn


# ----------------------------------------------------------
# Policy dispatch
# ----------------------------------------------------------

def get_remat_policy(
    policy_name: str,
):
    """
    Map config string -> JAX remat policy.
    """

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


# ----------------------------------------------------------
# Remat wrapper
# ----------------------------------------------------------

def remat_module(
    module_cls,
    *,
    policy: str,
    prevent_cse: bool = False,
):
    """
    Wrap module in nn.remat.

    Equivalent to MaxText/T5X rematerialization.
    """

    remat_policy = get_remat_policy(
        policy
    )

    return nn.remat(
        module_cls,
        policy=remat_policy,
        prevent_cse=prevent_cse,
    )
