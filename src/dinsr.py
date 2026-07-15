from __future__ import annotations

import ast
import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from backbone import TCMChainBaselineModel
except ModuleNotFoundError as exc:  # Allows rule parsing / CLI help without transformers installed.
    TCMChainBaselineModel = None
    _MODEL_IMPORT_ERROR = exc
else:
    _MODEL_IMPORT_ERROR = None


TASKS = ("diag", "syndrome", "treatment", "herb")
MULTICLASS_TASKS = {"diag", "syndrome"}
MULTILABEL_TASKS = {"treatment", "herb"}

SYMBOLIC_TYPES = ("token", "init_diag", "tcm_diag", "syndrome", "treatment", "herb")
TYPE_TO_CODE = {name: i for i, name in enumerate(SYMBOLIC_TYPES)}

TARGET_TYPES = ("tcm_diag", "syndrome", "treatment", "herb")
TARGET_TO_TASK = {
    "tcm_diag": "diag",
    "syndrome": "syndrome",
    "treatment": "treatment",
    "herb": "herb",
}
TASK_TO_TARGET = {task: target for target, task in TARGET_TO_TASK.items()}

NEGATIVE_USAGES = {
    "delete",
    "reject",
    "do_not_use",
    "exclude",
    "negative_evidence",
}

MODEL_VARIANTS = {"DiNSR": "dinsr"}


def _safe_read_csv(path: str | Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:  # pragma: no cover - best effort reader.
            last_error = exc
    raise ValueError(f"Could not read CSV file {path}: {last_error}")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "t"}


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int_id(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _parse_list_like(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
        except (SyntaxError, ValueError, TypeError):
            return []
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    return [text]


def _softplus_inverse(value: float) -> float:
    value = max(float(value), 1e-6)
    return math.log(math.expm1(value))


def _mapping_for_type(node_type: str, mappings: dict[str, Any], token_map: dict[int, int]) -> dict[int, int]:
    if node_type == "token":
        return token_map
    if node_type == "init_diag":
        return mappings["western_diag_map"]
    if node_type == "tcm_diag":
        return mappings["tcm_diag_map"]
    if node_type == "syndrome":
        return mappings["syndrome_map"]
    if node_type == "treatment":
        return mappings["treatment_map"]
    if node_type == "herb":
        return mappings["herb_map"]
    raise KeyError(f"Unknown node type: {node_type}")


def _size_for_type(node_type: str, mappings: dict[str, Any], token_map: dict[int, int]) -> int:
    return len(_mapping_for_type(node_type, mappings, token_map))


def _convert_node(
    node_type: str,
    node_id: Any,
    mappings: dict[str, Any],
    token_map: dict[int, int],
) -> int | None:
    original_id = _to_int_id(node_id)
    if original_id is None:
        return None
    return _mapping_for_type(node_type, mappings, token_map).get(original_id)


def _make_sparse_matrix(rows: list[tuple[int, int, float]], target_size: int, source_size: int) -> torch.Tensor:
    if not rows:
        indices = torch.empty((2, 0), dtype=torch.long)
        values = torch.empty((0,), dtype=torch.float32)
    else:
        indices = torch.tensor(
            [[target_idx for target_idx, _, _ in rows], [source_idx for _, source_idx, _ in rows]],
            dtype=torch.long,
        )
        values = torch.tensor([weight for _, _, weight in rows], dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=(target_size, source_size)).coalesce()


class LLMAuditedRuleWeightCalculator:
    """Converts LLM-audited rule metadata into a conservative numeric weight."""

    usage_weights = {
        "hard_constraint": 1.20,
        "hard_prior": 1.00,
        "strong_prior": 0.85,
        "soft_prior": 0.65,
        "moderate_prior": 0.45,
        "weak_prior": 0.25,
        "weak_hint": 0.15,
        "weak_signal": 0.10,
        "ranking_prior": 0.35,
        "auxiliary": 0.10,
        "context_dependent": 0.05,
        "context_only": 0.05,
    }
    risk_penalties = {
        "none": 1.00,
        "very_low": 1.00,
        "low": 0.90,
        "medium": 0.65,
        "high": 0.35,
        "very_high": 0.10,
    }
    task_weights = {
        "tcm_diag": 1.00,
        "syndrome": 0.90,
        "treatment": 1.00,
        "herb": 0.75,
    }

    def __init__(self, min_negative_weight: float = 0.03):
        self.min_negative_weight = min_negative_weight

    def is_negative(self, row: pd.Series) -> bool:
        usage = str(row.get("recommended_usage_final", "")).strip().lower()
        keep = _as_bool(row.get("keep", row.get("audit_keep", False)))
        final_adjustment = _to_float(row.get("final_weight_adjustment", 0.0), 0.0)
        return (not keep) or usage in NEGATIVE_USAGES or final_adjustment < 0

    def compute(self, row: pd.Series) -> float:
        target_type = str(row.get("consequent_type", "")).strip()
        usage = str(row.get("recommended_usage_final", "")).strip().lower()
        pseudo_risk = str(row.get("pseudo_correlation_risk_audit", "medium")).strip().lower()
        over_risk = str(row.get("overactivation_risk_audit", "medium")).strip().lower()

        confidence = _to_float(row.get("final_rule_confidence", row.get("llm_rule_confidence", 0.0)), 0.0)
        final_adjustment = _to_float(row.get("final_weight_adjustment", row.get("suggested_weight_scale", 0.0)), 0.0)
        adjustment = max(final_adjustment, 0.0)
        if self.is_negative(row):
            adjustment = max(abs(final_adjustment), self.min_negative_weight)

        quality = (
            confidence
            * adjustment
            * (_to_float(row.get("audit_score", 3.0), 3.0) / 5.0)
            * (_to_float(row.get("medical_consistency_score", 3.0), 3.0) / 5.0)
            * (_to_float(row.get("statistical_sanity_score", 3.0), 3.0) / 5.0)
            * (_to_float(row.get("specificity_audit_score", row.get("specificity_score", 3.0)), 3.0) / 5.0)
        )
        usage_weight = self.usage_weights.get(usage, 0.50)
        if usage in NEGATIVE_USAGES:
            usage_weight = 0.75
        risk_penalty = self.risk_penalties.get(pseudo_risk, 0.65) * self.risk_penalties.get(over_risk, 0.65)
        human_review_penalty = 0.70 if _as_bool(row.get("need_human_review_final", False)) else 1.0
        task_weight = self.task_weights.get(target_type, 0.80)
        return float(quality * usage_weight * risk_penalty * human_review_penalty * task_weight)


@dataclass
class LLMAuditedRuleIndex:
    matrices: dict[tuple[str, str, str], torch.Tensor] = field(default_factory=dict)
    rule_tensors: dict[tuple[str, str], dict[str, torch.Tensor]] = field(default_factory=dict)
    relation_keys: list[tuple[str, str, str]] = field(default_factory=list)
    source_sizes: dict[str, int] = field(default_factory=dict)
    target_sizes: dict[str, int] = field(default_factory=dict)
    rule_stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiGranularityRuleIndex:
    llm_index: LLMAuditedRuleIndex
    high_order_rules: dict[tuple[str, str], dict[str, torch.Tensor]] = field(default_factory=dict)
    path_rules: dict[tuple[str, str], dict[str, torch.Tensor]] = field(default_factory=dict)
    cooccurrence_matrices: dict[str, torch.Tensor] = field(default_factory=dict)
    rule_stats: dict[str, Any] = field(default_factory=dict)

    @property
    def source_sizes(self) -> dict[str, int]:
        return self.llm_index.source_sizes

    @property
    def target_sizes(self) -> dict[str, int]:
        return self.llm_index.target_sizes


def _existing_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None
    candidate = Path(text)
    return candidate if candidate.exists() else None


def _first_existing_column(row: pd.Series, names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if name in row and not pd.isna(row.get(name)):
            return row.get(name)
    return default


def _convert_node_loose(
    node_type: str,
    node_id: Any,
    mappings: dict[str, Any],
    token_map: dict[int, int],
) -> int | None:
    mapped = _convert_node(node_type, node_id, mappings, token_map)
    if mapped is not None:
        return mapped
    original_id = _to_int_id(node_id)
    if original_id is None:
        return None
    size = _size_for_type(node_type, mappings, token_map)
    if 0 <= original_id < size:
        return original_id
    return None


def _generic_rule_weight(row: pd.Series, default: float = 1.0) -> float:
    for name in (
        "final_weight",
        "weight",
        "final_rule_confidence",
        "confidence",
        "rule_score",
        "path_score_adjusted",
        "path_score",
        "smoothed_confidence",
        "lift",
        "pmi",
        "cooccur_freq",
        "cooccur_count",
    ):
        if name in row:
            value = _to_float(row.get(name), 0.0)
            if value > 0:
                return float(value)
    return float(default)


def _generic_rule_sign(row: pd.Series) -> str:
    usage = str(
        _first_existing_column(row, ("recommended_usage_final", "usage", "rule_tag", "pair_rule_level", "sign"), "")
    ).strip().lower()
    keep = _as_bool(row.get("keep", row.get("audit_keep", True)))
    final_adjustment = _to_float(row.get("final_weight_adjustment", 0.0), 0.0)
    if usage == "negative" or (not keep) or usage in NEGATIVE_USAGES or final_adjustment < 0:
        return "negative"
    return "positive"


def _pack_multi_antecedent_rows(
    rows: list[tuple[list[int], list[int], int, float]],
) -> dict[str, torch.Tensor]:
    if not rows:
        return {
            "antecedent_type_codes": torch.empty((0, 0), dtype=torch.long),
            "antecedent_indices": torch.empty((0, 0), dtype=torch.long),
            "antecedent_mask": torch.empty((0, 0), dtype=torch.bool),
            "target_indices": torch.empty((0,), dtype=torch.long),
            "weights": torch.empty((0,), dtype=torch.float32),
        }
    max_ants = max(len(type_codes) for type_codes, _, _, _ in rows)
    type_tensor = torch.zeros((len(rows), max_ants), dtype=torch.long)
    index_tensor = torch.zeros((len(rows), max_ants), dtype=torch.long)
    mask_tensor = torch.zeros((len(rows), max_ants), dtype=torch.bool)
    target_tensor = torch.empty((len(rows),), dtype=torch.long)
    weight_tensor = torch.empty((len(rows),), dtype=torch.float32)
    for i, (type_codes, indices, target_idx, weight) in enumerate(rows):
        width = len(type_codes)
        type_tensor[i, :width] = torch.tensor(type_codes, dtype=torch.long)
        index_tensor[i, :width] = torch.tensor(indices, dtype=torch.long)
        mask_tensor[i, :width] = True
        target_tensor[i] = int(target_idx)
        weight_tensor[i] = float(weight)
    return {
        "antecedent_type_codes": type_tensor,
        "antecedent_indices": index_tensor,
        "antecedent_mask": mask_tensor,
        "target_indices": target_tensor,
        "weights": weight_tensor,
    }


def _pad_rule_tensor(tensor: torch.Tensor, width: int, fill_value: int | bool = 0) -> torch.Tensor:
    if tensor.ndim != 2 or tensor.shape[1] >= width:
        return tensor
    pad_shape = (tensor.shape[0], width - tensor.shape[1])
    pad = torch.full(pad_shape, fill_value, dtype=tensor.dtype)
    return torch.cat([tensor, pad], dim=1)


def _merge_multi_rule_tensor_dicts(
    *rule_dicts: dict[tuple[str, str], dict[str, torch.Tensor]],
) -> dict[tuple[str, str], dict[str, torch.Tensor]]:
    merged: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    keys = set().union(*(rule_dict.keys() for rule_dict in rule_dicts))
    for key in keys:
        tensors = [rule_dict[key] for rule_dict in rule_dicts if key in rule_dict and rule_dict[key]["weights"].numel() > 0]
        if not tensors:
            continue
        width = max(t["antecedent_type_codes"].shape[1] for t in tensors)
        merged[key] = {
            "antecedent_type_codes": torch.cat(
                [_pad_rule_tensor(t["antecedent_type_codes"], width, 0) for t in tensors], dim=0
            ),
            "antecedent_indices": torch.cat(
                [_pad_rule_tensor(t["antecedent_indices"], width, 0) for t in tensors], dim=0
            ),
            "antecedent_mask": torch.cat(
                [_pad_rule_tensor(t["antecedent_mask"], width, False) for t in tensors], dim=0
            ),
            "target_indices": torch.cat([t["target_indices"] for t in tensors], dim=0),
            "weights": torch.cat([t["weights"] for t in tensors], dim=0),
        }
    return merged


def _build_multi_antecedent_rule_tensors_from_rows(
    parsed_rows: list[tuple[str, str, list[tuple[str, int]], int, float]],
) -> dict[tuple[str, str], dict[str, torch.Tensor]]:
    grouped: dict[tuple[str, str], list[tuple[list[int], list[int], int, float]]] = {}
    for target_type, sign, antecedents, target_idx, weight in parsed_rows:
        type_codes = [TYPE_TO_CODE[node_type] for node_type, _ in antecedents]
        indices = [idx for _, idx in antecedents]
        grouped.setdefault((target_type, sign), []).append((type_codes, indices, target_idx, weight))
    return {key: _pack_multi_antecedent_rows(rows) for key, rows in grouped.items()}


def build_high_order_rule_tensors(
    rule_path: str | Path,
    mappings: dict[str, Any],
    token_map: dict[int, int],
    min_antecedents: int = 1,
) -> tuple[dict[tuple[str, str], dict[str, torch.Tensor]], dict[str, Any]]:
    path = _existing_path(rule_path)
    if path is None:
        return {}, {"rules_loaded": 0, "rules_kept_positive": 0, "rules_kept_negative": 0, "missing_file": True}
    df = _safe_read_csv(path)
    parsed_rows: list[tuple[str, str, list[tuple[str, int]], int, float]] = []
    stats: dict[str, Any] = {
        "rules_loaded": int(len(df)),
        "rules_kept_positive": 0,
        "rules_kept_negative": 0,
        "rules_skipped_bad_target": 0,
        "rules_skipped_bad_antecedent": 0,
        "rules_skipped_mapping": 0,
        "rules_skipped_zero_weight": 0,
        "rules_skipped_antecedent_count": 0,
    }

    for _, row in df.iterrows():
        target_type = str(_first_existing_column(row, ("consequent_type", "target_type"), "")).strip()
        target_id = _first_existing_column(row, ("consequent_id", "target_id"), None)
        if target_type not in TARGET_TYPES:
            stats["rules_skipped_bad_target"] += 1
            continue
        target_idx = _convert_node_loose(target_type, target_id, mappings, token_map)
        if target_idx is None:
            stats["rules_skipped_mapping"] += 1
            continue
        ant_types = [str(x).strip() for x in _parse_list_like(row.get("antecedent_types"))]
        ant_ids = _parse_list_like(row.get("antecedent_ids"))
        if len(ant_types) != len(ant_ids) or not ant_types:
            stats["rules_skipped_bad_antecedent"] += 1
            continue
        if len(ant_types) < min_antecedents:
            stats["rules_skipped_antecedent_count"] += 1
            continue
        antecedents: list[tuple[str, int]] = []
        failed = False
        for ant_type, ant_id in zip(ant_types, ant_ids):
            if ant_type not in SYMBOLIC_TYPES:
                failed = True
                break
            ant_idx = _convert_node_loose(ant_type, ant_id, mappings, token_map)
            if ant_idx is None:
                failed = True
                break
            antecedents.append((ant_type, int(ant_idx)))
        if failed:
            stats["rules_skipped_mapping"] += 1
            continue
        weight = abs(_generic_rule_weight(row, default=1.0))
        if weight <= 0:
            stats["rules_skipped_zero_weight"] += 1
            continue
        sign = _generic_rule_sign(row)
        parsed_rows.append((target_type, sign, antecedents, int(target_idx), weight))
        stats["rules_kept_positive" if sign == "positive" else "rules_kept_negative"] += 1

    tensors = _build_multi_antecedent_rule_tensors_from_rows(parsed_rows)
    stats["relations"] = {f"{target}:{sign}": int(t["weights"].numel()) for (target, sign), t in tensors.items()}
    return tensors, stats


def _parse_path_nodes(value: Any) -> list[tuple[str, Any]]:
    text = str(value).strip() if value is not None and not pd.isna(value) else ""
    if not text:
        return []
    nodes: list[tuple[str, Any]] = []
    for part in text.split("->"):
        item = part.strip()
        if ":" not in item:
            continue
        node_type, node_id = item.split(":", 1)
        nodes.append((node_type.strip(), node_id.strip()))
    return nodes


def build_path_rule_tensors(
    rule_path: str | Path,
    mappings: dict[str, Any],
    token_map: dict[int, int],
) -> tuple[dict[tuple[str, str], dict[str, torch.Tensor]], dict[str, Any]]:
    path = _existing_path(rule_path)
    if path is None:
        return {}, {"rules_loaded": 0, "rules_kept_positive": 0, "rules_kept_negative": 0, "missing_file": True}
    df = _safe_read_csv(path)
    if {"antecedent_types", "antecedent_ids", "consequent_type", "consequent_id"}.issubset(df.columns):
        return build_high_order_rule_tensors(path, mappings, token_map)

    parsed_rows: list[tuple[str, str, list[tuple[str, int]], int, float]] = []
    stats: dict[str, Any] = {
        "rules_loaded": int(len(df)),
        "rules_kept_positive": 0,
        "rules_kept_negative": 0,
        "rules_skipped_bad_target": 0,
        "rules_skipped_bad_antecedent": 0,
        "rules_skipped_mapping": 0,
        "rules_skipped_zero_weight": 0,
    }
    for _, row in df.iterrows():
        target_type = str(_first_existing_column(row, ("target_type", "consequent_type"), "")).strip()
        target_id = _first_existing_column(row, ("target_id", "consequent_id"), None)
        if target_type not in TARGET_TYPES:
            stats["rules_skipped_bad_target"] += 1
            continue
        target_idx = _convert_node_loose(target_type, target_id, mappings, token_map)
        if target_idx is None:
            stats["rules_skipped_mapping"] += 1
            continue

        path_nodes = _parse_path_nodes(row.get("path_nodes"))
        if path_nodes:
            raw_antecedents = path_nodes[:-1]
        else:
            raw_antecedents = []
            source_type = str(row.get("source_type", "")).strip()
            if source_type:
                raw_antecedents.append((source_type, row.get("source_id")))
            mid_type = str(row.get("mid_type", "")).strip()
            if mid_type:
                raw_antecedents.append((mid_type, row.get("mid_id")))
        if not raw_antecedents:
            stats["rules_skipped_bad_antecedent"] += 1
            continue
        antecedents: list[tuple[str, int]] = []
        failed = False
        for ant_type, ant_id in raw_antecedents:
            if ant_type not in SYMBOLIC_TYPES:
                failed = True
                break
            ant_idx = _convert_node_loose(ant_type, ant_id, mappings, token_map)
            if ant_idx is None:
                failed = True
                break
            antecedents.append((ant_type, int(ant_idx)))
        if failed:
            stats["rules_skipped_mapping"] += 1
            continue
        weight = abs(_generic_rule_weight(row, default=1.0))
        if weight <= 0:
            stats["rules_skipped_zero_weight"] += 1
            continue
        sign = _generic_rule_sign(row)
        parsed_rows.append((target_type, sign, antecedents, int(target_idx), weight))
        stats["rules_kept_positive" if sign == "positive" else "rules_kept_negative"] += 1

    tensors = _build_multi_antecedent_rule_tensors_from_rows(parsed_rows)
    stats["relations"] = {f"{target}:{sign}": int(t["weights"].numel()) for (target, sign), t in tensors.items()}
    return tensors, stats


def build_cooccurrence_matrix(
    rule_path: str | Path,
    label_type: str,
    mappings: dict[str, Any],
    token_map: dict[int, int],
) -> tuple[torch.Tensor, dict[str, Any]]:
    path = _existing_path(rule_path)
    size = _size_for_type(label_type, mappings, token_map)
    empty = _make_sparse_matrix([], target_size=size, source_size=size)
    if path is None:
        return empty, {"rules_loaded": 0, "rules_kept": 0, "missing_file": True}
    df = _safe_read_csv(path)
    stats: dict[str, Any] = {
        "rules_loaded": int(len(df)),
        "rules_kept": 0,
        "rules_kept_pairs": 0,
        "matrix_edges": 0,
        "rules_skipped_mapping": 0,
        "rules_skipped_zero_weight": 0,
    }
    rows: list[tuple[int, int, float]] = []
    source_names = (
        ("source_treatment", "source_id", "treatment_a")
        if label_type == "treatment"
        else ("source_herb", "source_id", "herb_a")
    )
    target_names = (
        ("target_treatment", "target_id", "treatment_b")
        if label_type == "treatment"
        else ("target_herb", "target_id", "herb_b")
    )
    for _, row in df.iterrows():
        source_id = _first_existing_column(row, source_names, None)
        target_id = _first_existing_column(row, target_names, None)
        source_idx = _convert_node_loose(label_type, source_id, mappings, token_map)
        target_idx = _convert_node_loose(label_type, target_id, mappings, token_map)
        if source_idx is None or target_idx is None:
            stats["rules_skipped_mapping"] += 1
            continue
        weight = abs(_generic_rule_weight(row, default=1.0))
        if weight <= 0:
            stats["rules_skipped_zero_weight"] += 1
            continue
        rows.append((int(target_idx), int(source_idx), float(weight)))
        rows.append((int(source_idx), int(target_idx), float(weight)))
        stats["rules_kept"] += 1
        stats["rules_kept_pairs"] += 1
        stats["matrix_edges"] += 2
    return _make_sparse_matrix(rows, target_size=size, source_size=size), stats


def build_multi_granularity_rule_index(
    llm_rule_path: str | Path,
    mappings: dict[str, Any],
    token_map: dict[int, int],
    max_rules: int = 0,
    high_order_rule_path: str | Path | None = None,
    path_rule_path: str | Path | None = None,
    treatment_pair_rule_path: str | Path | None = None,
    herb_pair_rule_path: str | Path | None = None,
) -> MultiGranularityRuleIndex:
    llm_index = build_llm_audited_rule_index(llm_rule_path, mappings, token_map, max_rules=max_rules)
    audited_multi_rules, audited_multi_stats = build_high_order_rule_tensors(
        llm_rule_path,
        mappings,
        token_map,
        min_antecedents=2,
    )
    high_file_rules, high_stats = build_high_order_rule_tensors(high_order_rule_path or "", mappings, token_map)
    high_order_rules = _merge_multi_rule_tensor_dicts(audited_multi_rules, high_file_rules)
    path_rules, path_stats = build_path_rule_tensors(path_rule_path or "", mappings, token_map)
    treatment_matrix, treatment_stats = build_cooccurrence_matrix(
        treatment_pair_rule_path or "", "treatment", mappings, token_map
    )
    herb_matrix, herb_stats = build_cooccurrence_matrix(herb_pair_rule_path or "", "herb", mappings, token_map)
    cooccurrence_matrices = {
        "treatment": treatment_matrix,
        "herb": herb_matrix,
    }
    rule_stats = {
        **llm_index.rule_stats,
        "multi_granularity_enabled": True,
        "llm_first_order_rule_count": int(
            llm_index.rule_stats.get("rules_kept_positive", 0) + llm_index.rule_stats.get("rules_kept_negative", 0)
        ),
        "llm_multi_antecedent_rule_count": int(
            audited_multi_stats.get("rules_kept_positive", 0) + audited_multi_stats.get("rules_kept_negative", 0)
        ),
        "high_order_rule_count": int(
            audited_multi_stats.get("rules_kept_positive", 0)
            + audited_multi_stats.get("rules_kept_negative", 0)
            + high_stats.get("rules_kept_positive", 0)
            + high_stats.get("rules_kept_negative", 0)
        ),
        "path_rule_count": int(path_stats.get("rules_kept_positive", 0) + path_stats.get("rules_kept_negative", 0)),
        "treatment_pair_count": int(treatment_stats.get("rules_kept", 0)),
        "herb_pair_count": int(herb_stats.get("rules_kept", 0)),
        "llm_multi_antecedent_stats": audited_multi_stats,
        "high_order_stats": high_stats,
        "path_stats": path_stats,
        "treatment_pair_stats": treatment_stats,
        "herb_pair_stats": herb_stats,
    }
    return MultiGranularityRuleIndex(
        llm_index=llm_index,
        high_order_rules=high_order_rules,
        path_rules=path_rules,
        cooccurrence_matrices=cooccurrence_matrices,
        rule_stats=rule_stats,
    )


def build_llm_audited_rule_index(
    rule_path: str | Path,
    mappings: dict[str, Any],
    token_map: dict[int, int],
    max_rules: int = 0,
) -> LLMAuditedRuleIndex:
    df = _safe_read_csv(rule_path)
    if max_rules and max_rules > 0:
        df = df.head(max_rules).copy()

    required = {
        "antecedent_types",
        "antecedent_ids",
        "consequent_type",
        "consequent_id",
        "recommended_usage_final",
        "final_rule_confidence",
        "keep",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"LLM rule file is missing required columns: {missing}")

    source_sizes = {node_type: _size_for_type(node_type, mappings, token_map) for node_type in SYMBOLIC_TYPES}
    target_sizes = {target_type: source_sizes[target_type] for target_type in TARGET_TYPES}
    calculator = LLMAuditedRuleWeightCalculator()

    matrix_rows: dict[tuple[str, str, str], list[tuple[int, int, float]]] = {}
    rule_rows: dict[tuple[str, str], list[tuple[int, int, int, float]]] = {}
    stats: dict[str, Any] = {
        "rules_loaded": int(len(df)),
        "rules_kept_positive": 0,
        "rules_kept_negative": 0,
        "rules_skipped_bad_target": 0,
        "rules_skipped_bad_antecedent": 0,
        "rules_skipped_mapping": 0,
        "rules_skipped_zero_weight": 0,
        "rules_skipped_multi_antecedent": 0,
        "by_task": {},
        "by_usage": {},
        "by_risk": {},
    }

    for _, row in df.iterrows():
        target_type = str(row.get("consequent_type", "")).strip()
        if target_type not in TARGET_TYPES:
            stats["rules_skipped_bad_target"] += 1
            continue
        target_idx = _convert_node(target_type, row.get("consequent_id"), mappings, token_map)
        if target_idx is None:
            stats["rules_skipped_mapping"] += 1
            continue

        ant_types = [str(x).strip() for x in _parse_list_like(row.get("antecedent_types"))]
        ant_ids = _parse_list_like(row.get("antecedent_ids"))
        if len(ant_types) != len(ant_ids) or not ant_types:
            stats["rules_skipped_bad_antecedent"] += 1
            continue

        converted: list[tuple[str, int]] = []
        failed = False
        for ant_type, ant_id in zip(ant_types, ant_ids):
            if ant_type not in SYMBOLIC_TYPES:
                failed = True
                break
            ant_idx = _convert_node(ant_type, ant_id, mappings, token_map)
            if ant_idx is None:
                failed = True
                break
            converted.append((ant_type, int(ant_idx)))
        if failed:
            stats["rules_skipped_mapping"] += 1
            continue
        if len(converted) != 1:
            stats["rules_skipped_multi_antecedent"] += 1
            continue

        weight = calculator.compute(row)
        if weight <= 0:
            stats["rules_skipped_zero_weight"] += 1
            continue

        sign = "negative" if calculator.is_negative(row) else "positive"
        source_type, source_idx = converted[0]
        matrix_rows.setdefault((sign, source_type, target_type), []).append((target_idx, source_idx, weight))
        rule_rows.setdefault((target_type, sign), []).append(
            (TYPE_TO_CODE[source_type], source_idx, target_idx, weight)
        )

        if sign == "positive":
            stats["rules_kept_positive"] += 1
        else:
            stats["rules_kept_negative"] += 1
        task = str(row.get("task", TARGET_TO_TASK[target_type]))
        usage = str(row.get("recommended_usage_final", "")).strip().lower()
        risk = str(row.get("pseudo_correlation_risk_audit", "")).strip().lower()
        stats["by_task"][task] = stats["by_task"].get(task, 0) + 1
        stats["by_usage"][usage] = stats["by_usage"].get(usage, 0) + 1
        stats["by_risk"][risk] = stats["by_risk"].get(risk, 0) + 1

    matrices: dict[tuple[str, str, str], torch.Tensor] = {}
    relation_keys: list[tuple[str, str, str]] = []
    for key, rows in matrix_rows.items():
        sign, source_type, target_type = key
        matrices[key] = _make_sparse_matrix(
            rows,
            target_size=target_sizes[target_type],
            source_size=source_sizes[source_type],
        )
        relation_keys.append(key)

    rule_tensors: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    for key, rows in rule_rows.items():
        if rows:
            source_type_codes = torch.tensor([r[0] for r in rows], dtype=torch.long)
            source_indices = torch.tensor([r[1] for r in rows], dtype=torch.long)
            target_indices = torch.tensor([r[2] for r in rows], dtype=torch.long)
            weights = torch.tensor([r[3] for r in rows], dtype=torch.float32)
        else:
            source_type_codes = torch.empty((0,), dtype=torch.long)
            source_indices = torch.empty((0,), dtype=torch.long)
            target_indices = torch.empty((0,), dtype=torch.long)
            weights = torch.empty((0,), dtype=torch.float32)
        rule_tensors[key] = {
            "source_type_codes": source_type_codes,
            "source_indices": source_indices,
            "target_indices": target_indices,
            "weights": weights,
        }

    stats["relations"] = {f"{a}:{b}->{c}": int(matrices[(a, b, c)]._nnz()) for a, b, c in relation_keys}
    return LLMAuditedRuleIndex(
        matrices=matrices,
        rule_tensors=rule_tensors,
        relation_keys=relation_keys,
        source_sizes=source_sizes,
        target_sizes=target_sizes,
        rule_stats=stats,
    )


class LLMAuditedSymbolicReasoner(nn.Module):
    def __init__(self, rule_index: LLMAuditedRuleIndex):
        super().__init__()
        self.source_sizes = rule_index.source_sizes
        self.target_sizes = rule_index.target_sizes
        self.relation_specs: list[tuple[str, str, str, str]] = []
        self.relation_scales = nn.ParameterDict()

        for i, (sign, source_type, target_type) in enumerate(rule_index.relation_keys):
            name = f"rel_{i}"
            self.register_buffer(name, rule_index.matrices[(sign, source_type, target_type)].float().coalesce())
            self.relation_specs.append((sign, source_type, target_type, name))
            self.relation_scales[name] = nn.Parameter(torch.zeros(()))

        self.rule_specs: list[tuple[str, str, str]] = []
        for i, ((target_type, sign), tensors) in enumerate(rule_index.rule_tensors.items()):
            prefix = f"rules_{i}"
            self.rule_specs.append((target_type, sign, prefix))
            for key, tensor in tensors.items():
                self.register_buffer(f"{prefix}_{key}", tensor)

    def _zeros(self, batch_size: int, target_type: str, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.target_sizes[target_type], device=device)

    def compute_sum(
        self,
        target_type: str,
        activations: dict[str, torch.Tensor],
        use_relation_scales: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(iter(activations.values())).device
        batch_size = next(iter(activations.values())).shape[0]
        pos = self._zeros(batch_size, target_type, device)
        neg = self._zeros(batch_size, target_type, device)

        for sign, source_type, current_target, buffer_name in self.relation_specs:
            if current_target != target_type or source_type not in activations:
                continue
            matrix = getattr(self, buffer_name).float()
            source_activation = activations[source_type].float()
            device_type = source_activation.device.type
            autocast_context = (
                torch.amp.autocast(device_type=device_type, enabled=False)
                if device_type == "cuda"
                else nullcontext()
            )
            with autocast_context:
                contribution = torch.sparse.mm(matrix, source_activation.t()).t()
            if use_relation_scales:
                contribution = contribution * F.softplus(self.relation_scales[buffer_name])
            if sign == "positive":
                pos = pos + contribution
            else:
                neg = neg + contribution
        return pos, neg

    def compute_attention(
        self,
        target_type: str,
        activations: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(iter(activations.values())).device
        batch_size = next(iter(activations.values())).shape[0]
        pos = self._zeros(batch_size, target_type, device)
        neg = self._zeros(batch_size, target_type, device)

        for current_target, sign, prefix in self.rule_specs:
            if current_target != target_type:
                continue
            source_type_codes = getattr(self, f"{prefix}_source_type_codes")
            if source_type_codes.numel() == 0:
                continue
            source_indices = getattr(self, f"{prefix}_source_indices")
            target_indices = getattr(self, f"{prefix}_target_indices")
            weights = getattr(self, f"{prefix}_weights")

            numerator = self._zeros(batch_size, target_type, device)
            denominator = self._zeros(batch_size, target_type, device)
            for source_type, code in TYPE_TO_CODE.items():
                selected = source_type_codes == code
                if not bool(selected.any()) or source_type not in activations:
                    continue
                selected_source_idx = source_indices[selected]
                selected_target_idx = target_indices[selected]
                selected_weight = weights[selected].to(device)
                source_activation = activations[source_type].index_select(1, selected_source_idx)
                attention_mass = source_activation * selected_weight.abs().unsqueeze(0)
                contribution = source_activation * selected_weight.unsqueeze(0)
                expanded_targets = selected_target_idx.unsqueeze(0).expand(batch_size, -1)
                numerator.scatter_add_(1, expanded_targets, contribution)
                denominator.scatter_add_(1, expanded_targets, attention_mass)

            scores = numerator / denominator.clamp(min=1e-6)
            if sign == "positive":
                pos = pos + scores
            else:
                neg = neg + scores
        return pos, neg

    def forward(
        self,
        target_type: str,
        activations: dict[str, torch.Tensor],
        aggregation: str = "sum",
        use_relation_scales: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if aggregation == "attention":
            return self.compute_attention(target_type, activations)
        return self.compute_sum(target_type, activations, use_relation_scales=use_relation_scales)


class MultiGranularitySymbolicReasoner(nn.Module):
    """Combines first-order, high-order, path, and co-occurrence symbolic evidence."""

    source_names = ("first", "high", "path", "co")

    def __init__(self, rule_index: MultiGranularityRuleIndex, args: Any):
        super().__init__()
        self.source_sizes = rule_index.source_sizes
        self.target_sizes = rule_index.target_sizes
        self.first_order_reasoner = LLMAuditedSymbolicReasoner(rule_index.llm_index)
        self.first_order_available = bool(
            rule_index.llm_index.relation_keys or rule_index.llm_index.rule_tensors
        )
        self.high_order_activation = str(getattr(args, "high_order_activation", "soft_and")).lower()
        self.rule_source_attention = bool(getattr(args, "rule_source_attention", True))
        self.cooccurrence_weight_scale = float(getattr(args, "cooccurrence_weight_scale", 0.5))

        self.high_rule_specs: list[tuple[str, str, str]] = []
        for i, ((target_type, sign), tensors) in enumerate(rule_index.high_order_rules.items()):
            prefix = f"high_rules_{i}"
            self.high_rule_specs.append((target_type, sign, prefix))
            for key, tensor in tensors.items():
                self.register_buffer(f"{prefix}_{key}", tensor)

        self.path_rule_specs: list[tuple[str, str, str]] = []
        for i, ((target_type, sign), tensors) in enumerate(rule_index.path_rules.items()):
            prefix = f"path_rules_{i}"
            self.path_rule_specs.append((target_type, sign, prefix))
            for key, tensor in tensors.items():
                self.register_buffer(f"{prefix}_{key}", tensor)

        self.cooccurrence_specs: dict[str, str] = {}
        for label_type, matrix in rule_index.cooccurrence_matrices.items():
            buffer_name = f"cooccurrence_{label_type}"
            self.register_buffer(buffer_name, matrix.float().coalesce())
            self.cooccurrence_specs[label_type] = buffer_name

        hidden = int(getattr(args, "rule_source_attention_hidden_dim", 32))
        self.source_attention_mlps = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.Linear(len(self.source_names) * 5, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, len(self.source_names)),
                )
                for task in TASKS
            }
        )
        for mlp in self.source_attention_mlps.values():
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)

    def _zeros(self, batch_size: int, target_type: str, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.target_sizes[target_type], device=device)

    def _compute_multi_antecedent_rules(
        self,
        specs: list[tuple[str, str, str]],
        target_type: str,
        activations: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(iter(activations.values())).device
        batch_size = next(iter(activations.values())).shape[0]
        pos = self._zeros(batch_size, target_type, device)
        neg = self._zeros(batch_size, target_type, device)

        for current_target, sign, prefix in specs:
            if current_target != target_type:
                continue
            type_codes = getattr(self, f"{prefix}_antecedent_type_codes").to(device)
            if type_codes.numel() == 0:
                continue
            antecedent_indices = getattr(self, f"{prefix}_antecedent_indices").to(device)
            antecedent_mask = getattr(self, f"{prefix}_antecedent_mask").to(device)
            target_indices = getattr(self, f"{prefix}_target_indices").to(device)
            weights = getattr(self, f"{prefix}_weights").to(device)

            num_rules, max_ants = type_codes.shape
            values = torch.zeros(batch_size, num_rules, max_ants, device=device)
            observed = torch.zeros(num_rules, max_ants, dtype=torch.bool, device=device)
            for source_type, code in TYPE_TO_CODE.items():
                selected = (type_codes == code) & antecedent_mask
                if not bool(selected.any()) or source_type not in activations:
                    continue
                flat_indices = antecedent_indices[selected]
                gathered = activations[source_type].float().index_select(1, flat_indices)
                values[:, selected] = gathered
                observed[selected] = True

            expected_counts = antecedent_mask.sum(dim=1).float()
            observed_counts = observed.sum(dim=1).float()
            complete = (expected_counts > 0) & (observed_counts == expected_counts)
            valid = antecedent_mask.unsqueeze(0).float()
            denom = expected_counts.clamp(min=1.0).unsqueeze(0)
            if self.high_order_activation == "mean":
                trigger = (values * valid).sum(dim=2) / denom
            else:
                log_values = torch.log(values.clamp(min=1e-6)) * valid
                trigger = torch.exp(log_values.sum(dim=2) / denom)
            trigger = trigger * complete.float().unsqueeze(0)
            contribution = trigger * weights.unsqueeze(0)
            expanded_targets = target_indices.unsqueeze(0).expand(batch_size, -1)
            if sign == "positive":
                pos.scatter_add_(1, expanded_targets, contribution)
            else:
                neg.scatter_add_(1, expanded_targets, contribution)
        return pos, neg

    def _compute_cooccurrence(
        self,
        target_type: str,
        activations: dict[str, torch.Tensor],
        current_task_probs: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(iter(activations.values())).device
        batch_size = next(iter(activations.values())).shape[0]
        pos = self._zeros(batch_size, target_type, device)
        neg = self._zeros(batch_size, target_type, device)
        if target_type not in {"treatment", "herb"} or target_type not in self.cooccurrence_specs:
            return pos, neg

        source_probs = None
        if current_task_probs and target_type in current_task_probs:
            source_probs = current_task_probs[target_type]
        elif target_type in activations:
            source_probs = activations[target_type]
        if source_probs is None:
            return pos, neg
        matrix = getattr(self, self.cooccurrence_specs[target_type]).float()
        if matrix._nnz() == 0:
            return pos, neg
        source_probs = source_probs.float().to(device)
        contribution = torch.sparse.mm(matrix, source_probs.t()).t()
        pos = contribution * self.cooccurrence_weight_scale
        return pos, neg

    def _source_stats(self, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        evidence = pos - neg
        active = (evidence.abs() > 1e-8).float()
        return torch.stack(
            [
                active.mean(dim=1),
                evidence.abs().mean(dim=1),
                evidence.abs().max(dim=1).values,
                pos.clamp(min=0).mean(dim=1),
                neg.clamp(min=0).mean(dim=1),
            ],
            dim=1,
        )

    def _source_available(
        self,
        first_pos: torch.Tensor,
        first_neg: torch.Tensor,
        high_pos: torch.Tensor,
        high_neg: torch.Tensor,
        path_pos: torch.Tensor,
        path_neg: torch.Tensor,
        co_pos: torch.Tensor,
        co_neg: torch.Tensor,
    ) -> torch.Tensor:
        values = [
            self.first_order_available,
            bool((high_pos.abs().sum() + high_neg.abs().sum()).detach().item() > 0),
            bool((path_pos.abs().sum() + path_neg.abs().sum()).detach().item() > 0),
            bool((co_pos.abs().sum() + co_neg.abs().sum()).detach().item() > 0),
        ]
        return torch.tensor(values, dtype=torch.bool, device=first_pos.device)

    def _combine_sources(
        self,
        task: str,
        source_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos_stack = torch.stack([pair[0] for pair in source_pairs], dim=1)
        neg_stack = torch.stack([pair[1] for pair in source_pairs], dim=1)
        if not self.rule_source_attention:
            weights = pos_stack.new_ones(pos_stack.shape[0], len(source_pairs))
            pos = pos_stack.sum(dim=1)
            neg = neg_stack.sum(dim=1)
            return pos, neg, weights

        stats = torch.cat([self._source_stats(pos, neg) for pos, neg in source_pairs], dim=1)
        logits = self.source_attention_mlps[task](stats)
        available = self._source_available(
            source_pairs[0][0],
            source_pairs[0][1],
            source_pairs[1][0],
            source_pairs[1][1],
            source_pairs[2][0],
            source_pairs[2][1],
            source_pairs[3][0],
            source_pairs[3][1],
        )
        logits = logits.masked_fill(~available.unsqueeze(0), -1e4)
        weights = torch.softmax(logits, dim=1)
        pos = (weights.unsqueeze(-1) * pos_stack).sum(dim=1)
        neg = (weights.unsqueeze(-1) * neg_stack).sum(dim=1)
        return pos, neg, weights

    def forward(
        self,
        target_type: str,
        activations: dict[str, torch.Tensor],
        current_task_probs: dict[str, torch.Tensor] | None = None,
        base_repr: torch.Tensor | None = None,
        task: str | None = None,
        aggregation: str = "attention",
        use_relation_scales: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        task_name = task or TARGET_TO_TASK[target_type]
        first_pos, first_neg = self.first_order_reasoner(
            target_type,
            activations,
            aggregation=aggregation,
            use_relation_scales=use_relation_scales,
        )
        high_pos, high_neg = self._compute_multi_antecedent_rules(self.high_rule_specs, target_type, activations)
        path_pos, path_neg = self._compute_multi_antecedent_rules(self.path_rule_specs, target_type, activations)
        co_pos, co_neg = self._compute_cooccurrence(target_type, activations, current_task_probs)
        source_pairs = [(first_pos, first_neg), (high_pos, high_neg), (path_pos, path_neg), (co_pos, co_neg)]
        pos, neg, source_weights = self._combine_sources(task_name, source_pairs)
        details = {
            "source_names": self.source_names,
            "source_weights": source_weights,
            "first_pos": first_pos,
            "first_neg": first_neg,
            "high_pos": high_pos,
            "high_neg": high_neg,
            "path_pos": path_pos,
            "path_neg": path_neg,
            "co_pos": co_pos,
            "co_neg": co_neg,
            "first_norm": (first_pos - first_neg).abs().mean(dim=1),
            "high_norm": (high_pos - high_neg).abs().mean(dim=1),
            "path_norm": (path_pos - path_neg).abs().mean(dim=1),
            "co_norm": (co_pos - co_neg).abs().mean(dim=1),
        }
        return pos, neg, details


class LowRankLabelMLP(nn.Module):
    def __init__(self, num_labels: int, dropout: float):
        super().__init__()
        hidden = max(16, min(256, num_labels // 2 if num_labels >= 32 else num_labels))
        self.net = nn.Sequential(
            nn.LayerNorm(num_labels),
            nn.Linear(num_labels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.net(logits)


class DiNSR(nn.Module):
    def __init__(
        self,
        args: Any,
        mappings: dict[str, Any],
        token_map: dict[int, int],
        rule_index: LLMAuditedRuleIndex | MultiGranularityRuleIndex,
    ):
        super().__init__()
        self.args = args
        self.mappings = mappings
        self.token_map = token_map
        if TCMChainBaselineModel is None:
            raise ModuleNotFoundError(
                "TCMChainBaselineModel could not be imported. Install the project neural dependencies, "
                "especially `transformers`, before training."
            ) from _MODEL_IMPORT_ERROR
        self.neural = TCMChainBaselineModel(args, mappings)
        self.uses_multi_granularity_rules = isinstance(rule_index, MultiGranularityRuleIndex)
        if self.uses_multi_granularity_rules:
            self.reasoner = MultiGranularitySymbolicReasoner(rule_index, args)
        else:
            self.reasoner = LLMAuditedSymbolicReasoner(rule_index)

        self.num_tcm_diag = self.neural.num_tcm_diag
        self.num_syndrome = self.neural.num_syndrome
        self.num_treatment = self.neural.num_treatment
        self.num_herb = self.neural.num_herb
        self.num_token = len(token_map)
        self.num_init_diag = len(mappings["western_diag_map"])

        task_dims = {
            "diag": self.num_tcm_diag,
            "syndrome": self.num_syndrome,
            "treatment": self.num_treatment,
            "herb": self.num_herb,
        }
        self.symbolic_norms = nn.ModuleDict({task: nn.LayerNorm(dim) for task, dim in task_dims.items()})
        self.label_mlps = nn.ModuleDict(
            {task: LowRankLabelMLP(dim, getattr(args, "rule_dropout", 0.10)) for task, dim in task_dims.items()}
        )
        gamma_defaults = {
            "diag": getattr(args, "fusion_gamma_diag", 0.06),
            "syndrome": getattr(args, "fusion_gamma_syndrome", 0.08),
            "treatment": getattr(args, "fusion_gamma_treatment", 0.12),
            "herb": getattr(args, "fusion_gamma_herb", 0.06),
        }
        self.fusion_gamma_raw = nn.ParameterDict(
            {
                task: nn.Parameter(torch.tensor(_softplus_inverse(gamma_defaults[task]), dtype=torch.float32))
                for task in TASKS
            }
        )

        gate_hidden = int(getattr(args, "fusion_gate_hidden_dim", 32))
        self.gate_mlps = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.Linear(8, gate_hidden),
                    nn.GELU(),
                    nn.Dropout(getattr(args, "rule_dropout", 0.10)),
                    nn.Linear(gate_hidden, 1),
                )
                for task in TASKS
            }
        )

    def _one_hot(self, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        return F.one_hot(labels.clamp(min=0), num_classes=num_classes).float()

    def _make_initial_activations(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        device = batch["input_ids"].device
        batch_size = batch["input_ids"].shape[0]
        token_activation = torch.zeros(batch_size, self.num_token, device=device)
        if self.num_token > 0 and "token_ids" in batch:
            token_ids = batch["token_ids"]
            token_mask = batch.get("token_mask", token_ids > 0)
            source_idx = (token_ids - 1).clamp(min=0)
            values = token_mask.float()
            token_activation.scatter_add_(1, source_idx, values)
            token_activation = token_activation.clamp(max=1.0)

        init_activation = torch.zeros(batch_size, self.num_init_diag, device=device)
        init_diag = batch.get("init_diag", batch["western_diag"]).clamp(min=0)
        init_activation.scatter_(1, init_diag.unsqueeze(1), 1.0)
        return {"token": token_activation, "init_diag": init_activation}

    def _task_probs(self, task: str, logits: torch.Tensor) -> torch.Tensor:
        if task in MULTICLASS_TASKS:
            return torch.softmax(logits, dim=1)
        return torch.sigmoid(logits)

    def _confidence(self, task: str, probs: torch.Tensor) -> torch.Tensor:
        if task in MULTICLASS_TASKS:
            return probs.max(dim=1).values
        top_k = min(5, probs.shape[1])
        return torch.topk(probs, top_k, dim=1).values.mean(dim=1)

    def _activation_from_logits_or_labels(
        self,
        task: str,
        logits: torch.Tensor,
        batch: dict[str, torch.Tensor],
        mode: str,
    ) -> torch.Tensor:
        if task == "diag":
            if mode in {"train", "oracle"}:
                return self._one_hot(batch["tcm_diag"], self.num_tcm_diag)
            return torch.softmax(logits, dim=1)
        if task == "syndrome":
            if mode in {"train", "oracle"}:
                return self._one_hot(batch["syndrome"], self.num_syndrome)
            return torch.softmax(logits, dim=1)
        if task == "treatment":
            if mode in {"train", "oracle"}:
                return batch["treatment"].float()
            return torch.sigmoid(logits)
        raise ValueError(f"Unsupported activation task: {task}")

    def _chain_embedding(self, logits: torch.Tensor, embedding_layer: nn.Embedding) -> torch.Tensor:
        return self.neural.get_chain_embedding(logits, embedding_layer)

    def _disagreement(self, task: str, neural_logits: torch.Tensor, symbolic_logits: torch.Tensor) -> torch.Tensor:
        neural_probs = self._task_probs(task, neural_logits)
        symbolic_probs = self._task_probs(task, symbolic_logits)
        if task in MULTICLASS_TASKS:
            return 0.5 * torch.abs(neural_probs - symbolic_probs).sum(dim=1).clamp(max=1.0)
        return torch.abs(neural_probs - symbolic_probs).mean(dim=1).clamp(max=1.0)

    def _feature_vector(
        self,
        task: str,
        neural_logits: torch.Tensor,
        symbolic_logits: torch.Tensor,
        pos_logits: torch.Tensor,
        neg_logits: torch.Tensor,
    ) -> torch.Tensor:
        neural_probs = self._task_probs(task, neural_logits)
        symbolic_probs = self._task_probs(task, symbolic_logits)
        disagreement = self._disagreement(task, neural_logits, symbolic_logits)
        active = ((pos_logits.abs() + neg_logits.abs()) > 1e-8).float()
        coverage = active.mean(dim=1)
        pos_mean = pos_logits.clamp(min=0).mean(dim=1)
        neg_mean = neg_logits.clamp(min=0).mean(dim=1)
        gap = (pos_mean - neg_mean).tanh()
        entropy = -(symbolic_probs.clamp(min=1e-6) * symbolic_probs.clamp(min=1e-6).log()).mean(dim=1)
        entropy = entropy / max(math.log(max(symbolic_probs.shape[1], 2)), 1e-6)
        return torch.stack(
            [
                self._confidence(task, neural_probs),
                self._confidence(task, symbolic_probs),
                1.0 - disagreement,
                disagreement,
                coverage,
                entropy.clamp(max=1.0),
                gap,
                (pos_logits.abs().mean(dim=1) + neg_logits.abs().mean(dim=1)).tanh(),
            ],
            dim=1,
        )

    def _symbolic_component(
        self,
        task: str,
        activations: dict[str, torch.Tensor],
        base_repr: torch.Tensor,
        current_task_probs: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        target_type = TASK_TO_TARGET[task]
        reasoner_outputs = self.reasoner(
            target_type,
            activations,
            current_task_probs=current_task_probs,
            base_repr=base_repr,
            task=task,
            aggregation="attention",
            use_relation_scales=True,
        ) if self.uses_multi_granularity_rules else self.reasoner(
            target_type,
            activations,
            aggregation="attention",
            use_relation_scales=True,
        )
        if len(reasoner_outputs) == 3:
            pos, neg, rule_details = reasoner_outputs
        else:
            pos, neg = reasoner_outputs
            rule_details = {}
        raw = pos - neg
        raw = torch.tanh(raw) + 0.5 * self.label_mlps[task](raw)
        return raw, pos, neg, rule_details

    def _fuse(
        self,
        task: str,
        neural_logits: torch.Tensor,
        symbolic_logits: torch.Tensor,
        pos_logits: torch.Tensor,
        neg_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma = F.softplus(self.fusion_gamma_raw[task])
        symbolic_component = self.symbolic_norms[task](symbolic_logits)
        features = self._feature_vector(task, neural_logits, symbolic_component, pos_logits, neg_logits)
        disagreement = features[:, 3]

        learned_gate = torch.sigmoid(self.gate_mlps[task](features))
        conflict_factor = (1.0 - disagreement).unsqueeze(1)
        gate = learned_gate * conflict_factor
        fused = neural_logits + gate * gamma * symbolic_component

        return fused, gate, disagreement

    def _forward_chain(self, batch: dict[str, torch.Tensor], mode: str) -> dict[str, Any]:
        activations = self._make_initial_activations(batch)
        base_repr = self.neural.get_base_representation(batch)

        neural_outputs: dict[str, torch.Tensor] = {}
        fused_outputs: dict[str, torch.Tensor] = {}
        symbolic_outputs: dict[str, torch.Tensor] = {}
        symbolic_pos: dict[str, torch.Tensor] = {}
        symbolic_neg: dict[str, torch.Tensor] = {}
        gates: dict[str, torch.Tensor] = {}
        disagreements: dict[str, torch.Tensor] = {}
        rule_details: dict[str, Any] = {}
        detach_rule_probs = bool(getattr(self.args, "detach_cooccurrence_probs", True))

        diag_hidden = self.neural.diag_mlp(base_repr)
        diag_logits = self.neural.diag_head(diag_hidden)
        neural_outputs["diag"] = diag_logits
        sym, pos, neg, details = self._symbolic_component("diag", activations, base_repr)
        rule_details["diag"] = details
        symbolic_outputs["diag"], symbolic_pos["diag"], symbolic_neg["diag"] = sym, pos, neg
        fused_diag, gates["diag"], disagreements["diag"] = self._fuse(
            "diag", diag_logits, sym, pos, neg
        )
        fused_outputs["diag"] = fused_diag
        activations["tcm_diag"] = self._activation_from_logits_or_labels("diag", fused_diag, batch, mode)

        if mode in {"train", "oracle"}:
            diag_emb = self.neural.tcm_diag_embedding(batch["tcm_diag"])
        else:
            diag_emb = self._chain_embedding(fused_diag, self.neural.tcm_diag_embedding)

        syndrome_input = torch.cat([base_repr, diag_hidden, diag_emb], dim=1)
        syndrome_hidden = self.neural.syndrome_mlp(syndrome_input)
        syndrome_logits = self.neural.syndrome_head(syndrome_hidden)
        neural_outputs["syndrome"] = syndrome_logits
        sym, pos, neg, details = self._symbolic_component("syndrome", activations, base_repr)
        rule_details["syndrome"] = details
        symbolic_outputs["syndrome"], symbolic_pos["syndrome"], symbolic_neg["syndrome"] = sym, pos, neg
        fused_syndrome, gates["syndrome"], disagreements["syndrome"] = self._fuse(
            "syndrome", syndrome_logits, sym, pos, neg
        )
        fused_outputs["syndrome"] = fused_syndrome
        activations["syndrome"] = self._activation_from_logits_or_labels("syndrome", fused_syndrome, batch, mode)

        if mode in {"train", "oracle"}:
            syndrome_emb = self.neural.syndrome_embedding(batch["syndrome"])
        else:
            syndrome_emb = self._chain_embedding(fused_syndrome, self.neural.syndrome_embedding)

        treatment_input = torch.cat([base_repr, diag_hidden, syndrome_hidden, diag_emb, syndrome_emb], dim=1)
        treatment_hidden = self.neural.treatment_mlp(treatment_input)
        treatment_logits = self.neural.treatment_head(treatment_hidden)
        neural_outputs["treatment"] = treatment_logits
        treatment_rule_probs = torch.sigmoid(treatment_logits)
        if detach_rule_probs:
            treatment_rule_probs = treatment_rule_probs.detach()
        sym, pos, neg, details = self._symbolic_component(
            "treatment",
            activations,
            base_repr,
            current_task_probs={"treatment": treatment_rule_probs},
        )
        rule_details["treatment"] = details
        symbolic_outputs["treatment"], symbolic_pos["treatment"], symbolic_neg["treatment"] = sym, pos, neg
        fused_treatment, gates["treatment"], disagreements["treatment"] = self._fuse(
            "treatment", treatment_logits, sym, pos, neg
        )
        fused_outputs["treatment"] = fused_treatment
        activations["treatment"] = self._activation_from_logits_or_labels("treatment", fused_treatment, batch, mode)

        if mode in {"train", "oracle"}:
            treatment_repr = self.neural.treatment_projection(batch["treatment"])
        else:
            treatment_repr = self.neural.treatment_projection(torch.sigmoid(fused_treatment))

        herb_input = torch.cat(
            [base_repr, diag_hidden, syndrome_hidden, treatment_hidden, diag_emb, syndrome_emb, treatment_repr],
            dim=1,
        )
        herb_hidden = self.neural.herb_mlp(herb_input)
        herb_logits = self.neural.herb_head(herb_hidden)
        neural_outputs["herb"] = herb_logits
        herb_rule_probs = torch.sigmoid(herb_logits)
        treatment_rule_probs = torch.sigmoid(fused_treatment)
        if detach_rule_probs:
            herb_rule_probs = herb_rule_probs.detach()
            treatment_rule_probs = treatment_rule_probs.detach()
        sym, pos, neg, details = self._symbolic_component(
            "herb",
            activations,
            base_repr,
            current_task_probs={"herb": herb_rule_probs, "treatment": treatment_rule_probs},
        )
        rule_details["herb"] = details
        symbolic_outputs["herb"], symbolic_pos["herb"], symbolic_neg["herb"] = sym, pos, neg
        fused_herb, gates["herb"], disagreements["herb"] = self._fuse(
            "herb", herb_logits, sym, pos, neg
        )
        fused_outputs["herb"] = fused_herb

        return {
            "fused": fused_outputs,
            "neural": neural_outputs,
            "symbolic": symbolic_outputs,
            "symbolic_pos": symbolic_pos,
            "symbolic_neg": symbolic_neg,
            "gates": gates,
            "disagreements": disagreements,
            "rule_details": rule_details,
        }

    def forward(self, batch: dict[str, torch.Tensor], mode: str = "train") -> dict[str, Any]:
        if mode not in {"train", "oracle", "predict"}:
            raise ValueError(f"Invalid mode: {mode}")
        return self._forward_chain(batch, mode)


def create_dinsr_model(
    args: Any,
    mappings: dict[str, Any],
    token_map: dict[int, int],
    rule_index: LLMAuditedRuleIndex | MultiGranularityRuleIndex,
    model_name: str,
) -> DiNSR:
    if model_name not in MODEL_VARIANTS:
        valid = ", ".join(MODEL_VARIANTS)
        raise ValueError(f"Unknown model_name={model_name}. Valid options: {valid}")
    return DiNSR(
        args=args,
        mappings=mappings,
        token_map=token_map,
        rule_index=rule_index,
    )
