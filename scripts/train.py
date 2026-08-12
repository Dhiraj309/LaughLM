from __future__ import annotations

from huggingface_hub import hf_hub_download
import jax

from LaughLM.config.loader import load_config
from LaughLM.utils.data_factory import create_dataloader
from LaughLM.data.memmap_loader import MemmapDataset
from LaughLM.training.trainer import Trainer
from LaughLM.training.fsdp_trainer import FSDPTrainer


TRAINER_REGISTRY = {
    "pmap": Trainer,
    "fsdp": FSDPTrainer,
}


RESERVED_BACKENDS = {
    "parallel3d": (
        "runtime.backend='parallel3d' is reserved, but "
        "Parallel3DTrainer is not implemented yet."
    ),
    "moe": (
        "runtime.backend='moe' is reserved, but "
        "MoETrainer is not implemented yet."
    ),
}


def resolve_backend(config) -> str:
    """
    Return canonical backend name.

    Backward compatibility:
      gspmd -> fsdp
    """

    return getattr(
        config.runtime,
        "canonical_backend",
        config.runtime.backend,
    )


def resolve_trainer_class(config):
    """
    Resolve trainer class from runtime.backend.

    PMAP and FSDP are implemented.
    Parallel3D and MoE are intentionally reserved and fail clearly.
    """

    backend = resolve_backend(config)

    if backend in RESERVED_BACKENDS:
        raise NotImplementedError(
            RESERVED_BACKENDS[backend]
        )

    try:
        return TRAINER_REGISTRY[backend]
    except KeyError as e:
        raise ValueError(
            "Unknown runtime.backend.\n"
            f"  raw backend:       {config.runtime.backend!r}\n"
            f"  canonical backend: {backend!r}\n"
            f"  available:         {sorted(TRAINER_REGISTRY)}\n"
            f"  reserved:          {sorted(RESERVED_BACKENDS)}"
        ) from e


def resolve_data_replicas(config) -> int:
    """
    Resolve the data-parallel replica count used by the input pipeline.

    PMAP:
      use actual JAX device count.

    FSDP:
      use configured mesh data axis.
    """

    backend = resolve_backend(config)

    if backend == "pmap":
        return int(jax.device_count())

    if backend == "fsdp":
        return int(
            config.spmd.mesh.axis_sizes()["data"]
        )

    if backend in RESERVED_BACKENDS:
        raise NotImplementedError(
            RESERVED_BACKENDS[backend]
        )

    raise ValueError(
        f"Cannot resolve data replicas for backend={backend!r}"
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")

    config = load_config(args.config)

    raw_backend = config.runtime.backend
    backend = resolve_backend(config)

    if raw_backend != backend:
        print(
            "[train] runtime.backend alias:\n"
            f"  raw={raw_backend!r}\n"
            f"  canonical={backend!r}",
            flush=True,
        )

    trainer_cls = resolve_trainer_class(config)
    data_replicas = resolve_data_replicas(config)

    files = [
        "fineweb-edu/fineweb-edu_shard_00008.bin",
        "fineweb-edu/fineweb-edu_shard_00009.bin",
        "fineweb-edu/fineweb-edu_shard_00010.bin",
        "fineweb-edu/fineweb-edu_shard_00011.bin",
    ]

    paths = [
        hf_hub_download(
            repo_id="LaughTaleAI/LaughLM-Tokenized",
            filename=f,
            repo_type="dataset",
        )
        for f in files
    ]

    dataset = create_dataloader(
        config=config,
        paths=paths,
        global_batch_size=(
            config.runtime.micro_batch_per_device
            * data_replicas
        ),
        process_index=jax.process_index(),
        process_count=jax.process_count(),
    )

    trainer = trainer_cls(
        config=config,
        resume_dir=config.runtime.checkpoint_dir,
    )

    trainer.train(dataset)


if __name__ == "__main__":
    main()
