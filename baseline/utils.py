from __future__ import annotations

import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

from .models import CooccurrenceGraphData, GraphRelationData, NODE_TYPES, TASKS
from dataset import parse_herb_dict_keys, parse_list_field
from metrics import compute_multiclass_metrics, compute_multilabel_metrics
from symbolic_dataset import parse_token_field


LABEL_KEYS = {
    "diag": "tcm_diag",
    "syndrome": "syndrome",
    "treatment": "treatment",
    "herb": "herb",
}


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def save_json(path: str | Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_builtin(payload), handle, ensure_ascii=False, indent=2)


def compute_chain_score(metrics: dict[str, dict[str, float]]) -> float:
    return float(sum(metrics[task]["score"] for task in TASKS) / len(TASKS))


def flatten_metrics(prefix: str, metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for task, task_metrics in metrics.items():
        for metric_name, value in task_metrics.items():
            flattened[f"{prefix}_{task}_{metric_name}"] = value
    flattened[f"{prefix}_chain_score"] = compute_chain_score(metrics)
    return flattened


def resolve_rank_k(configured_k: int, labels: np.ndarray) -> int:
    if configured_k is not None and configured_k > 0:
        return int(configured_k)
    average_count = int(round(float(np.mean(np.sum(labels, axis=1)))))
    return max(average_count, 1)


def compute_metrics_from_arrays(
    preds: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    mappings: dict[str, Any],
    args: Any,
) -> dict[str, dict[str, float]]:
    return {
        "diag": compute_multiclass_metrics(
            preds["diag"],
            labels["diag"],
            len(mappings["tcm_diag_map"]),
            top_k=int(args.diag_top_k),
        ),
        "syndrome": compute_multiclass_metrics(
            preds["syndrome"],
            labels["syndrome"],
            len(mappings["syndrome_map"]),
            top_k=int(args.syndrome_top_k),
        ),
        "treatment": compute_multilabel_metrics(
            preds["treatment"],
            labels["treatment"],
            threshold=float(args.multilabel_threshold),
            rank_k=resolve_rank_k(int(args.treatment_top_k), labels["treatment"]),
        ),
        "herb": compute_multilabel_metrics(
            preds["herb"],
            labels["herb"],
            threshold=float(args.multilabel_threshold),
            rank_k=resolve_rank_k(int(args.herb_top_k), labels["herb"]),
        ),
    }


def print_score_lines(prefix: str, metrics: dict[str, dict[str, float]]) -> None:
    print(
        f"{prefix}: "
        f"diag={metrics['diag']['score']:.4f}, "
        f"syndrome={metrics['syndrome']['score']:.4f}, "
        f"treatment={metrics['treatment']['score']:.4f}, "
        f"herb={metrics['herb']['score']:.4f}, "
        f"chain={compute_chain_score(metrics):.4f}"
    )


def print_epoch_summary(
    epoch: int,
    args: Any,
    train_loss: float,
    oracle_loss: float,
    oracle_metrics: dict[str, dict[str, float]],
    predicted_loss: float,
    predicted_metrics: dict[str, dict[str, float]],
) -> None:
    print(
        f"Epoch {epoch}/{args.epochs}: "
        f"train_loss={train_loss:.4f}, "
        f"valid_oracle_loss={oracle_loss:.4f}, "
        f"valid_predicted_loss={predicted_loss:.4f}, "
        f"valid_predicted_chain_score={compute_chain_score(predicted_metrics):.4f}"
    )
    print_score_lines("  Oracle", oracle_metrics)
    print_score_lines("  Predicted", predicted_metrics)


def print_metrics_table(name: str, metrics: dict[str, dict[str, float]]) -> None:
    print(f"\n========== {name} Results ==========")
    print(f"Chain Score: {compute_chain_score(metrics):.4f}")
    for task in TASKS:
        print(f"\n[{task}]")
        print(pd.DataFrame([metrics[task]]).to_string(index=False))
    print("=======================================")


def _map_single(value: Any, mapping: dict[int, int]) -> int:
    try:
        return int(mapping.get(int(float(value)), 0))
    except (TypeError, ValueError):
        return 0


def _text_value(row: pd.Series, args: Any) -> str:
    text = row.get(args.text_col, "")
    if pd.isna(text) or not str(text).strip():
        text = f"{row.get('主诉', '')}{row.get('简要病史', '')}{row.get('体格检查', '')}"
    return str(text)


class TCMComparisonDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Any | None,
        mappings: dict[str, Any],
        token_map: dict[int, int],
        args: Any,
        include_text: bool,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.mappings = mappings
        self.token_map = token_map
        self.args = args
        self.include_text = include_text

    def __len__(self) -> int:
        return len(self.df)

    def _age(self, row: pd.Series) -> torch.Tensor:
        try:
            value = float(row[self.args.age_col])
        except (TypeError, ValueError):
            value = float(getattr(self.args, "age_mean", 0.0))
        mean = float(getattr(self.args, "age_mean", 0.0))
        std = float(getattr(self.args, "age_std", 1.0))
        return torch.tensor((value - mean) / max(std, 1e-6), dtype=torch.float32)

    def _sex(self, row: pd.Series) -> torch.Tensor:
        try:
            value = int(float(row[self.args.sex_col]))
        except (TypeError, ValueError):
            value = 0
        return torch.tensor(0 if value <= 0 else 1, dtype=torch.long)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        item: dict[str, torch.Tensor] = {
            "age": self._age(row),
            "sex": self._sex(row),
            "western_diag": torch.tensor(
                _map_single(row[self.args.western_diag_col], self.mappings["western_diag_map"]),
                dtype=torch.long,
            ),
            "tcm_diag": torch.tensor(_map_single(row[self.args.tcm_diag_col], self.mappings["tcm_diag_map"]), dtype=torch.long),
            "syndrome": torch.tensor(_map_single(row[self.args.syndrome_col], self.mappings["syndrome_map"]), dtype=torch.long),
        }

        treatment_target = torch.zeros(len(self.mappings["treatment_map"]), dtype=torch.float32)
        for label in parse_list_field(row[self.args.treatment_col]):
            mapped = self.mappings["treatment_map"].get(int(label))
            if mapped is not None:
                treatment_target[mapped] = 1.0
        item["treatment"] = treatment_target

        herb_target = torch.zeros(len(self.mappings["herb_map"]), dtype=torch.float32)
        for label in parse_herb_dict_keys(row[self.args.herb_col]):
            mapped = self.mappings["herb_map"].get(int(label))
            if mapped is not None:
                herb_target[mapped] = 1.0
        item["herb"] = herb_target

        token_indices = []
        for token_id in parse_token_field(row[self.args.token_col]):
            mapped = self.token_map.get(int(token_id))
            if mapped is not None:
                token_indices.append(mapped + 1)
        item["token_ids"] = torch.tensor(sorted(set(token_indices)), dtype=torch.long)

        if self.include_text:
            if self.tokenizer is None:
                raise ValueError("Text comparison dataset requires a tokenizer.")
            encoded = self.tokenizer(
                _text_value(row, self.args),
                max_length=int(self.args.max_length),
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            item["input_ids"] = encoded["input_ids"].squeeze(0)
            item["attention_mask"] = encoded["attention_mask"].squeeze(0)
        return item


def comparison_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_tokens = max((len(item["token_ids"]) for item in batch), default=0)
    max_tokens = max(max_tokens, 1)
    token_ids = torch.zeros(len(batch), max_tokens, dtype=torch.long)
    token_mask = torch.zeros(len(batch), max_tokens, dtype=torch.bool)
    for row_index, item in enumerate(batch):
        length = len(item["token_ids"])
        if length:
            token_ids[row_index, :length] = item["token_ids"]
            token_mask[row_index, :length] = True
        else:
            token_mask[row_index, 0] = True

    collated = {
        key: torch.stack([item[key] for item in batch])
        for key in batch[0]
        if key not in {"token_ids"}
    }
    collated["token_ids"] = token_ids
    collated["token_mask"] = token_mask
    return collated


def dataframe_label_arrays(df: pd.DataFrame, mappings: dict[str, Any], args: Any) -> dict[str, np.ndarray]:
    diag = np.asarray([_map_single(value, mappings["tcm_diag_map"]) for value in df[args.tcm_diag_col]], dtype=np.int64)
    syndrome = np.asarray([_map_single(value, mappings["syndrome_map"]) for value in df[args.syndrome_col]], dtype=np.int64)
    treatment = np.zeros((len(df), len(mappings["treatment_map"])), dtype=np.float32)
    herb = np.zeros((len(df), len(mappings["herb_map"])), dtype=np.float32)
    for row_index, value in enumerate(df[args.treatment_col]):
        for label in parse_list_field(value):
            mapped = mappings["treatment_map"].get(int(label))
            if mapped is not None:
                treatment[row_index, mapped] = 1.0
    for row_index, value in enumerate(df[args.herb_col]):
        for label in parse_herb_dict_keys(value):
            mapped = mappings["herb_map"].get(int(label))
            if mapped is not None:
                herb[row_index, mapped] = 1.0
    return {"diag": diag, "syndrome": syndrome, "treatment": treatment, "herb": herb}


def _compute_multiclass_weights(series: pd.Series, label_map: dict[int, int], max_weight: float) -> torch.Tensor:
    counts = torch.zeros(len(label_map), dtype=torch.float32)
    for label in series.dropna():
        mapped = label_map.get(int(label))
        if mapped is not None:
            counts[mapped] += 1
    weights = torch.ones_like(counts)
    nonzero = counts > 0
    if nonzero.any():
        weights[nonzero] = counts[nonzero].sum() / (nonzero.sum().float() * counts[nonzero])
    return weights.clamp(max=max_weight)


def _compute_multilabel_pos_weight(
    series: pd.Series,
    label_map: dict[int, int],
    parser: Any,
    max_weight: float,
) -> torch.Tensor:
    positives = torch.zeros(len(label_map), dtype=torch.float32)
    for labels in series.dropna().apply(parser):
        for label in labels:
            mapped = label_map.get(int(label))
            if mapped is not None:
                positives[mapped] += 1
    negatives = float(len(series)) - positives
    return (negatives / positives.clamp(min=1.0)).clamp(min=1.0, max=max_weight)


def attach_training_statistics(args: Any, train_df: pd.DataFrame, mappings: dict[str, Any]) -> None:
    age_series = pd.to_numeric(train_df[args.age_col], errors="coerce")
    age_mean = float(age_series.mean()) if not age_series.dropna().empty else 0.0
    age_std = float(age_series.std()) if not age_series.dropna().empty else 1.0
    args.age_mean = age_mean
    args.age_std = age_std if age_std > 1e-6 else 1.0
    if getattr(args, "disable_loss_weights", False):
        args.diag_class_weights = None
        args.syndrome_class_weights = None
        args.treatment_pos_weight = None
        args.herb_pos_weight = None
        return
    args.diag_class_weights = _compute_multiclass_weights(train_df[args.tcm_diag_col], mappings["tcm_diag_map"], args.max_class_weight)
    args.syndrome_class_weights = _compute_multiclass_weights(
        train_df[args.syndrome_col], mappings["syndrome_map"], args.max_class_weight
    )
    args.treatment_pos_weight = _compute_multilabel_pos_weight(
        train_df[args.treatment_col],
        mappings["treatment_map"],
        parse_list_field,
        args.max_pos_weight,
    )
    args.herb_pos_weight = _compute_multilabel_pos_weight(
        train_df[args.herb_col],
        mappings["herb_map"],
        parse_herb_dict_keys,
        args.max_pos_weight,
    )


def make_loss_functions(args: Any, device: torch.device):
    diag_weight = args.diag_class_weights.to(device) if args.diag_class_weights is not None else None
    syndrome_weight = args.syndrome_class_weights.to(device) if args.syndrome_class_weights is not None else None
    treatment_weight = args.treatment_pos_weight.to(device) if args.treatment_pos_weight is not None else None
    herb_weight = args.herb_pos_weight.to(device) if args.herb_pos_weight is not None else None
    return (
        nn.CrossEntropyLoss(weight=diag_weight),
        nn.CrossEntropyLoss(weight=syndrome_weight),
        nn.BCEWithLogitsLoss(pos_weight=treatment_weight),
        nn.BCEWithLogitsLoss(pos_weight=herb_weight),
    )


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def output_tuple_to_dict(outputs: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
    return {task: output for task, output in zip(TASKS, outputs)}


def compute_task_loss(
    logits: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    loss_fns: tuple[nn.Module, nn.Module, nn.Module, nn.Module],
    args: Any,
) -> torch.Tensor:
    diag_loss, syndrome_loss, treatment_loss, herb_loss = loss_fns
    return (
        args.lambda_diag * diag_loss(logits["diag"], batch["tcm_diag"])
        + args.lambda_syndrome * syndrome_loss(logits["syndrome"], batch["syndrome"])
        + args.lambda_treatment * treatment_loss(logits["treatment"], batch["treatment"])
        + args.lambda_herb * herb_loss(logits["herb"], batch["herb"])
    )


def train_torch_one_epoch(
    model: nn.Module,
    dataloader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    loss_fns: tuple[nn.Module, nn.Module, nn.Module, nn.Module],
    args: Any,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(dataloader, desc="Training", leave=False)

    def optimizer_step() -> None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer_was_stepped = (not scaler.is_enabled()) or scaler.get_scale() >= scale_before
        if scheduler is not None and optimizer_was_stepped:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(progress):
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast("cuda", enabled=bool(args.use_amp and device.type == "cuda")):
            logits = output_tuple_to_dict(model(batch, mode="train"))
            loss = compute_task_loss(logits, batch, loss_fns, args)
            scaled_loss = loss / int(args.gradient_accumulation_steps)
        scaler.scale(scaled_loss).backward()
        if (step + 1) % int(args.gradient_accumulation_steps) == 0:
            optimizer_step()
        total_loss += float(loss.detach().item())
        progress.set_postfix(loss=total_loss / (step + 1))
    if len(dataloader) % int(args.gradient_accumulation_steps) != 0:
        optimizer_step()
    return total_loss / max(len(dataloader), 1)


def evaluate_torch_chain(
    model: nn.Module,
    dataloader: Any,
    device: torch.device,
    loss_fns: tuple[nn.Module, nn.Module, nn.Module, nn.Module],
    mappings: dict[str, Any],
    args: Any,
    mode: str,
) -> tuple[float, dict[str, dict[str, float]]]:
    model.eval()
    total_loss = 0.0
    preds = {task: [] for task in TASKS}
    labels = {task: [] for task in TASKS}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating ({mode})", leave=False):
            batch = move_batch_to_device(batch, device)
            with torch.amp.autocast("cuda", enabled=bool(args.use_amp and device.type == "cuda")):
                logits = output_tuple_to_dict(model(batch, mode=mode))
                loss = compute_task_loss(logits, batch, loss_fns, args)
            total_loss += float(loss.item())
            for task in TASKS:
                preds[task].append(logits[task].detach().cpu().numpy())
                labels[task].append(batch[LABEL_KEYS[task]].detach().cpu().numpy())
    pred_arrays = {task: np.concatenate(values, axis=0) for task, values in preds.items()}
    label_arrays = {task: np.concatenate(values, axis=0) for task, values in labels.items()}
    return total_loss / max(len(dataloader), 1), compute_metrics_from_arrays(pred_arrays, label_arrays, mappings, args)


def _mapped_tokens(row: pd.Series, token_map: dict[int, int], args: Any) -> list[int]:
    return sorted({token_map[token] for token in parse_token_field(row[args.token_col]) if token in token_map})


def _mapped_multilabels(value: Any, mapping: dict[int, int], parser: Any) -> list[int]:
    labels = []
    for label in parser(value):
        mapped = mapping.get(int(label))
        if mapped is not None:
            labels.append(int(mapped))
    return sorted(set(labels))


def _counter_edges(counter: Counter, top_k_per_source: int, min_count: int):
    by_source: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (source, target), count in counter.items():
        if count >= min_count:
            by_source[int(source)].append((int(target), int(count)))
    for source, edges in by_source.items():
        edges.sort(key=lambda item: item[1], reverse=True)
        for target, count in edges[:top_k_per_source]:
            yield source, target, count


def _finalize_relation(
    name: str,
    source_type: str,
    target_type: str,
    counter: Counter,
    top_k_per_source: int,
    min_count: int,
) -> GraphRelationData:
    rows = list(_counter_edges(counter, top_k_per_source, min_count))
    if not rows:
        return GraphRelationData(
            name=name,
            source_type=source_type,
            target_type=target_type,
            source_index=torch.zeros(0, dtype=torch.long),
            target_index=torch.zeros(0, dtype=torch.long),
            weight=torch.zeros(0, dtype=torch.float32),
        )
    source, target, count = zip(*rows)
    weight = np.log1p(np.asarray(count, dtype=np.float32))
    weight = weight / max(float(weight.max()), 1.0)
    return GraphRelationData(
        name=name,
        source_type=source_type,
        target_type=target_type,
        source_index=torch.tensor(source, dtype=torch.long),
        target_index=torch.tensor(target, dtype=torch.long),
        weight=torch.tensor(weight, dtype=torch.float32),
    )


def build_train_cooccurrence_graph(
    train_df: pd.DataFrame,
    mappings: dict[str, Any],
    token_map: dict[int, int],
    args: Any,
) -> CooccurrenceGraphData:
    relation_meta: dict[str, tuple[str, str]] = {}
    counters: dict[str, Counter] = defaultdict(Counter)

    def add_relation(name: str, source_type: str, target_type: str, sources: list[int], targets: list[int]) -> None:
        if not sources or not targets:
            return
        relation_meta[name] = (source_type, target_type)
        for source in sources:
            for target in targets:
                counters[name][(int(source), int(target))] += 1

    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Building train cooccurrence graph", leave=False):
        tokens = _mapped_tokens(row, token_map, args)[: int(args.graph_max_tokens_per_case)]
        western = [_map_single(row[args.western_diag_col], mappings["western_diag_map"])]
        diag = [_map_single(row[args.tcm_diag_col], mappings["tcm_diag_map"])]
        syndrome = [_map_single(row[args.syndrome_col], mappings["syndrome_map"])]
        treatment = _mapped_multilabels(row[args.treatment_col], mappings["treatment_map"], parse_list_field)
        herb = _mapped_multilabels(row[args.herb_col], mappings["herb_map"], parse_herb_dict_keys)

        add_relation("western_diag_to_token", "western_diag", "token", western, tokens)
        for target_name, target_ids in (("diag", diag), ("syndrome", syndrome), ("treatment", treatment), ("herb", herb)):
            add_relation(f"token_to_{target_name}", "token", target_name, tokens, target_ids)
            add_relation(f"western_diag_to_{target_name}", "western_diag", target_name, western, target_ids)
        add_relation("diag_to_syndrome", "diag", "syndrome", diag, syndrome)
        add_relation("diag_to_treatment", "diag", "treatment", diag, treatment)
        add_relation("diag_to_herb", "diag", "herb", diag, herb)
        add_relation("syndrome_to_treatment", "syndrome", "treatment", syndrome, treatment)
        add_relation("syndrome_to_herb", "syndrome", "herb", syndrome, herb)
        add_relation("treatment_to_herb", "treatment", "herb", treatment, herb)

        for source, target in combinations(treatment, 2):
            add_relation("treatment_to_treatment", "treatment", "treatment", [source], [target])
            add_relation("treatment_to_treatment", "treatment", "treatment", [target], [source])
        for source, target in combinations(herb, 2):
            add_relation("herb_to_herb", "herb", "herb", [source], [target])
            add_relation("herb_to_herb", "herb", "herb", [target], [source])

    relations = []
    for name, counter in counters.items():
        source_type, target_type = relation_meta[name]
        relation = _finalize_relation(
            name,
            source_type,
            target_type,
            counter,
            int(args.graph_top_k_per_source),
            int(args.graph_min_edge_count),
        )
        relations.append(relation)
        if source_type != target_type:
            reverse_counter = Counter({(target, source): count for (source, target), count in counter.items()})
            relations.append(
                _finalize_relation(
                    f"{target_type}_to_{source_type}_reverse_{name}",
                    target_type,
                    source_type,
                    reverse_counter,
                    int(args.graph_top_k_per_source),
                    int(args.graph_min_edge_count),
                )
            )

    node_sizes = {
        "token": len(token_map),
        "western_diag": len(mappings["western_diag_map"]),
        "diag": len(mappings["tcm_diag_map"]),
        "syndrome": len(mappings["syndrome_map"]),
        "treatment": len(mappings["treatment_map"]),
        "herb": len(mappings["herb_map"]),
    }
    stats = {
        "source": "train_split_dataframe",
        "node_sizes": node_sizes,
        "relations": {
            relation.name: {
                "source_type": relation.source_type,
                "target_type": relation.target_type,
                "edges": int(relation.source_index.numel()),
            }
            for relation in relations
        },
        "top_k_per_source": int(args.graph_top_k_per_source),
        "min_edge_count": int(args.graph_min_edge_count),
    }
    if any(node_sizes[node_type] <= 0 for node_type in NODE_TYPES):
        raise ValueError(f"Invalid cooccurrence graph node sizes: {node_sizes}")
    return CooccurrenceGraphData(node_sizes=node_sizes, relations=relations, stats=stats)
