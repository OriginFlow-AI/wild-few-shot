from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(
    experiment: str | Path,
    *,
    base: str | Path | None = None,
) -> dict[str, Any]:
    experiment_path = Path(experiment).resolve()
    if base is None:
        base_path = experiment_path.parents[1] / "base.yaml"
    else:
        base_path = Path(base).resolve()
    with base_path.open(encoding="utf-8") as handle:
        base_cfg = yaml.safe_load(handle)
    with experiment_path.open(encoding="utf-8") as handle:
        experiment_cfg = yaml.safe_load(handle)
    cfg = _deep_merge(base_cfg, experiment_cfg)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    required = (
        cfg["run"]["name"],
        cfg["model"]["type"],
        cfg["optimizer"]["lr"],
        cfg["trainer"]["per_rank_batch"],
        cfg["trainer"]["accumulate_grad_batches"],
    )
    if any(value is None for value in required):
        raise ValueError("config contains unresolved required null values")
    if bool(cfg["data"].get("require_force_weight", False)) and not cfg["paths"].get(
        "train_supervision_root"
    ):
        raise ValueError("require_force_weight=true requires paths.train_supervision_root")
    if bool(cfg["data"].get("require_binary_force_weight", False)) and not bool(
        cfg["data"].get("require_force_weight", False)
    ):
        raise ValueError("require_binary_force_weight=true requires require_force_weight=true")
    model_type = str(cfg["model"]["type"])
    if model_type not in {
        "tds_lstm",
        "k2_direct",
        "k2_affine",
        "k2_force_moe",
        "k2_spectral_residual",
        "k2_force15_refine_direct",
        "k2_force15_refine_affine",
    }:
        raise ValueError(f"unknown model.type: {model_type}")
    aggregation = cfg.get("validation", {}).get("aggregation", "pooled")
    if cfg.get("warm_start"):
        warm_start = cfg["warm_start"]
        if cfg.get("transfer"):
            raise ValueError("warm_start and transfer are mutually exclusive")
        if model_type != "k2_direct":
            raise ValueError("full-model warm_start currently requires k2_direct")
        if not warm_start.get("checkpoint"):
            raise ValueError("warm_start requires a checkpoint")
        if not isinstance(warm_start.get("renormalize_force_head", True), bool):
            raise ValueError("warm_start.renormalize_force_head must be boolean")
    if cfg.get("transfer"):
        transfer = cfg["transfer"]
        if model_type not in {"k2_direct", "k2_affine"}:
            raise ValueError("encoder transfer requires k2_direct or k2_affine")
        if not transfer.get("checkpoint") or int(transfer["freeze_epochs"]) < 0:
            raise ValueError("transfer requires a checkpoint and nonnegative freeze_epochs")
        if not 0 < float(transfer["encoder_lr"]) <= float(cfg["optimizer"]["lr"]):
            raise ValueError("encoder_lr must be positive and no greater than decoder lr")
        if int(cfg["trainer"]["devices"]) != 1:
            raise ValueError("staged encoder transfer currently supports single-device training")
    if aggregation not in {"pooled", "per_trial_macro"}:
        raise ValueError(f"unknown validation aggregation: {aggregation}")
    if model_type.startswith("k2_force15_"):
        if aggregation != "pooled":
            raise ValueError("Force15 comparison protocol selects by pooled Val force loss")
        if not cfg["model"].get("expert_artifact"):
            raise ValueError("Force15 requires an expert artifact")
        if float(cfg["model"].get("refinement_penalty_weight", 0.001)) < 0:
            raise ValueError("negative refinement penalty")
    output = (
        int(cfg["data"]["window_length"])
        - int(cfg["data"]["left_context"])
        - int(cfg["data"]["right_context"])
    )
    if output != 10_000:
        raise ValueError(f"expected exactly 10000 scored samples, got {output}")


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)
