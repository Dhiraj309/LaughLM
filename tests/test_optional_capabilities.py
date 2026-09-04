"""Optional Gen-2 capability checks.

Tokamax is intentionally optional for the stable lane.  These tests therefore
skip when the distribution is not installed and validate only its public
surface when it is present.
"""

import importlib
import importlib.metadata

import pytest


TOKAMAX_PUBLIC_CAPABILITIES = (
    "linear_softmax_cross_entropy_loss",
    "dot_product_attention",
    "ragged_dot",
)


def test_tokamax_public_capabilities_are_optional() -> None:
    try:
        importlib.metadata.version("tokamax")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("Tokamax is an optional Gen-2 dependency")

    tokamax = importlib.import_module("tokamax")
    missing = [
        name
        for name in TOKAMAX_PUBLIC_CAPABILITIES
        if not hasattr(tokamax, name)
    ]
    assert not missing, f"Tokamax public capabilities are missing: {missing}"
