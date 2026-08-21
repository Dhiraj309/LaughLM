"""Shared LaughLM dataset-artifact contract helpers."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Mapping

CONTRACT_NAME = "laughlm_dataset_artifact"
CONTRACT_VERSION = 1
VALID_ARTIFACT_TYPES = {"dataset_stage", "training_run"}
VALID_STAGES = {"stage1", "stage2", "stage3", "stage4", "training"}


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_artifact_contract(
    *, artifact_type: str, stage: str, dataset_id: str, run_id: str,
    config_hash: str, source_refs: Iterable[Mapping[str, Any]] = (),
    attributes: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    contract = {
        "name": CONTRACT_NAME, "version": CONTRACT_VERSION,
        "artifact_type": str(artifact_type), "stage": str(stage),
        "dataset_id": str(dataset_id), "run_id": str(run_id),
        "config_hash": str(config_hash),
        "source_refs": [dict(ref) for ref in source_refs],
        "attributes": dict(attributes or {}),
    }
    require_valid_artifact_contract(contract)
    return contract


def validate_artifact_contract(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["artifact_contract must be an object"]
    errors: list[str] = []
    if value.get("name") != CONTRACT_NAME:
        errors.append(f"name must be {CONTRACT_NAME!r}")
    if value.get("version") != CONTRACT_VERSION:
        errors.append(f"version must be {CONTRACT_VERSION}")
    if value.get("artifact_type") not in VALID_ARTIFACT_TYPES:
        errors.append(f"artifact_type must be one of {sorted(VALID_ARTIFACT_TYPES)}")
    if value.get("stage") not in VALID_STAGES:
        errors.append(f"stage must be one of {sorted(VALID_STAGES)}")
    for field in ("dataset_id", "run_id", "config_hash"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            errors.append(f"{field} must be a non-empty string")
    if not isinstance(value.get("source_refs"), list):
        errors.append("source_refs must be a list")
    elif any(not isinstance(ref, dict) for ref in value["source_refs"]):
        errors.append("source_refs entries must be objects")
    if not isinstance(value.get("attributes"), dict):
        errors.append("attributes must be an object")
    return errors


def require_valid_artifact_contract(value: Any) -> None:
    errors = validate_artifact_contract(value)
    if errors:
        raise ValueError("Invalid artifact contract: " + "; ".join(errors))
