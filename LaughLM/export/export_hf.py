"""
LaughLM/export/export_hf.py

Canonical Hugging Face export pipeline for LaughLM.

Pipeline
--------
1. Restore checkpoint
2. Extract params
3. Convert tensors
4. Build HF config
5. Save safetensors
6. Save config.json
7. Save generation_config.json
8. Copy tokenizer assets
9. Validate export

Design goals
------------
- deterministic
- scan-compatible
- tied-embedding aware
- TPU-safe
- safetensors-native
- HF ecosystem compatible
"""

from __future__ import annotations

import json
import shutil

from pathlib import Path

from safetensors.numpy import save_file

from transformers import (
    GenerationConfig,
)

from LaughLM.config.loader import (
    load_config,
)

from LaughLM.model.llama.config_factory import (
    build_llama_config,
)

from LaughLM.training.checkpoint import (
    CheckpointManager,
)

from LaughLM.training.train_state import (
    TrainState,
)

from LaughLM.export.convert_params import (
    convert_params_to_hf,
)

from LaughLM.export.hf_config import (
    build_hf_config,
)

from LaughLM.export.validate_hf import (
    validate_hf_export,
)


# ============================================================
# Tokenizer copy
# ============================================================


def copy_tokenizer_files(
    source_dir,
    output_dir,
):
    """
    Copy tokenizer assets into HF export directory.
    """

    source_dir = Path(source_dir)

    output_dir = Path(output_dir)

    tokenizer_files = [

        "tokenizer.json",

        "tokenizer.model",

        "tokenizer_config.json",

        "special_tokens_map.json",

        "added_tokens.json",
    ]

    copied = []

    for filename in tokenizer_files:

        src = source_dir / filename

        if src.exists():

            dst = output_dir / filename

            shutil.copy2(
                src,
                dst,
            )

            copied.append(filename)

    if not copied:

        raise RuntimeError(
            "No tokenizer files found.\n"
            f"source_dir={source_dir}"
        )

    print(
        "[export] copied tokenizer files:"
    )

    for filename in copied:

        print(f"  - {filename}")


# ============================================================
# Generation config
# ============================================================


def save_generation_config(
    output_dir,
    llama_config,
):
    """
    Save generation_config.json
    """

    generation_config = (
        GenerationConfig(

            bos_token_id=(
                llama_config.bos_token_id
            ),

            eos_token_id=(
                llama_config.eos_token_id
            ),

            pad_token_id=(
                llama_config.pad_token_id
            ),

            max_length=(
                llama_config.max_position_embeddings
            ),

            do_sample=False,

            use_cache=True,
        )
    )

    generation_config.save_pretrained(
        output_dir
    )

    print(
        "[export] saved generation config"
    )


# ============================================================
# Config save
# ============================================================


def save_hf_config(
    output_dir,
    hf_config,
):
    """
    Save HF config.json
    """

    output_dir = Path(output_dir)

    config_path = (
        output_dir
        / "config.json"
    )

    with open(
        config_path,
        "w",
    ) as f:

        json.dump(
            hf_config,
            f,
            indent=2,
        )

    print(
        "[export] saved config.json"
    )


# ============================================================
# Main export
# ============================================================


def export_hf_checkpoint(
    *,
    config_path,
    checkpoint_dir,
    output_dir,
    tokenizer_dir,
    validate=True,
):
    """
    Export LaughLM checkpoint to Hugging Face format.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Load experiment config
    # ========================================================

    print(
        "[export] loading config..."
    )

    exp_config = load_config(
        config_path
    )

    llama_config = build_llama_config(
        exp_config
    )

    # ========================================================
    # Restore checkpoint
    # ========================================================

    print(
        "[export] restoring checkpoint..."
    )

    checkpoints = CheckpointManager(
        checkpoint_dir
    )

    restored = (
        checkpoints.restore_latest(
            target_state=None,
        )
    )

    if restored is None:

        raise RuntimeError(
            "No checkpoint found."
        )

    state, step = restored

    print(
        f"[export] restored "
        f"step={step:,}"
    )

    # ========================================================
    # Extract params
    # ========================================================

    if isinstance(
        state,
        TrainState,
    ):

        params = state.params

    else:

        params = state["params"]

    # ========================================================
    # Convert params
    # ========================================================

    print(
        "[export] converting tensors..."
    )

    tensors = convert_params_to_hf(
        params=params,
        config=llama_config,
    )

    total_tensors = len(tensors)

    total_params = sum(
        tensor.size
        for tensor in tensors.values()
    )

    print(
        f"[export] converted "
        f"{total_tensors:,} tensors"
    )

    print(
        f"[export] total params: "
        f"{total_params:,}"
    )

    # ========================================================
    # Save safetensors
    # ========================================================

    print(
        "[export] saving safetensors..."
    )

    safetensor_path = (
        output_dir
        / "model.safetensors"
    )

    metadata = {

        "format": "pt",

        "framework": "huggingface",

        "source": "LaughLM",
    }

    save_file(
        tensors,
        str(safetensor_path),
        metadata=metadata,
    )

    print(
        "[export] saved model.safetensors"
    )

    # ========================================================
    # HF config
    # ========================================================

    print(
        "[export] building HF config..."
    )

    hf_config = build_hf_config(
        llama_config
    )

    save_hf_config(
        output_dir,
        hf_config,
    )

    # ========================================================
    # Generation config
    # ========================================================

    save_generation_config(
        output_dir,
        llama_config,
    )

    # ========================================================
    # Tokenizer
    # ========================================================

    print(
        "[export] copying tokenizer..."
    )

    copy_tokenizer_files(
        tokenizer_dir,
        output_dir,
    )

    # ========================================================
    # Validation
    # ========================================================

    if validate:

        print(
            "[export] running validation..."
        )

        validate_hf_export(
            hf_dir=output_dir,
            config_path=config_path,
            params=params,
        )

    print(
        "\n[export] COMPLETE"
    )

    print(
        f"[export] output dir:\n"
        f"{output_dir}"
    )


# ============================================================
# CLI
# ============================================================


def main():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--checkpoint_dir",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        required=True,
    )

    parser.add_argument(
        "--tokenizer_dir",
        required=True,
    )

    parser.add_argument(
        "--skip_validation",
        action="store_true",
    )

    args = parser.parse_args()

    export_hf_checkpoint(

        config_path=(
            args.config
        ),

        checkpoint_dir=(
            args.checkpoint_dir
        ),

        output_dir=(
            args.output_dir
        ),

        tokenizer_dir=(
            args.tokenizer_dir
        ),

        validate=(
            not args.skip_validation
        ),
    )


if __name__ == "__main__":

    main()
