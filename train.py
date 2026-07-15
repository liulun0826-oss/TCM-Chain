from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import build_label_mappings, parse_herb_dict_keys, parse_list_field
from dinsr import (
    MODEL_VARIANTS,
    MULTILABEL_TASKS,
    TASKS,
    build_llm_audited_rule_index,
    build_multi_granularity_rule_index,
    create_dinsr_model,
)
from metrics import compute_multiclass_metrics, compute_multilabel_metrics
from neural_symbolic_dataset import TCMNeuralSymbolicDataset, neural_symbolic_collate_fn
from symbolic_dataset import build_token_map
from utils import get_device, print_gpu_info, read_csv_with_encoding_fallback, set_seed


LABEL_KEYS = {
    "diag": "tcm_diag",
    "syndrome": "syndrome",
    "treatment": "treatment",
    "herb": "herb",
}


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(payload), f, ensure_ascii=False, indent=2)


def compute_chain_score(metrics: dict[str, dict[str, float]]) -> float:
    return float(
        (
            metrics["diag"]["score"]
            + metrics["syndrome"]["score"]
            + metrics["treatment"]["score"]
            + metrics["herb"]["score"]
        )
        / 4.0
    )


def flatten_metrics(prefix: str, metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    row: dict[str, float] = {}
    for task, task_metrics in metrics.items():
        for key, value in task_metrics.items():
            row[f"{prefix}_{task}_{key}"] = value
    row[f"{prefix}_chain_score"] = compute_chain_score(metrics)
    return row


def protocol_dict(args: argparse.Namespace, rule_stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_family": "LLM-audited neural-symbolic comparison",
        "checkpoint_selection": "valid_predicted_chain_score",
        "chain_score": "mean(diag_macro_f1, syndrome_macro_f1, treatment_micro_f1, herb_micro_f1)",
        "multiclass_primary": "macro_f1",
        "multilabel_primary": "micro_f1 at fixed threshold",
        "multilabel_threshold": args.multilabel_threshold,
        "diag_top_k": args.diag_top_k,
        "syndrome_top_k": args.syndrome_top_k,
        "treatment_rank_k": args.treatment_top_k,
        "herb_rank_k": args.herb_top_k,
        "chain_inference": args.chain_inference,
        "label_mapping_mode": args.label_mapping_mode,
        "rule_source": args.llm_rule_path,
        "enable_multi_granularity_rules": args.enable_multi_granularity_rules,
        "high_order_rule_path": args.high_order_rule_path,
        "path_rule_path": args.path_rule_path,
        "treatment_pair_rule_path": args.treatment_pair_rule_path,
        "herb_pair_rule_path": args.herb_pair_rule_path,
        "rule_source_attention": args.rule_source_attention,
        "rule_source_attention_hidden_dim": args.rule_source_attention_hidden_dim,
        "high_order_activation": args.high_order_activation,
        "cooccurrence_weight_scale": args.cooccurrence_weight_scale,
        "detach_cooccurrence_probs": args.detach_cooccurrence_probs,
        "fusion_mode": getattr(args, "fusion_mode", "learned_gate"),
        "ablation_variant": getattr(args, "ablation_variant", "Full Model"),
        "symbolic_train_mode": "oracle",
        "symbolic_predict_mode": "predict_soft",
        "rule_consistency_tasks": args.rule_consistency_tasks,
        "rule_consistency_source": args.rule_consistency_source,
        "beta_rule": args.beta_rule,
        "rule_warmup_epochs": args.rule_warmup_epochs,
        "rule_consistency_threshold": args.rule_consistency_threshold,
        "alpha_neural": args.alpha_neural,
        "alpha_symbolic": args.alpha_symbolic,
        "rules_loaded": rule_stats.get("rules_loaded"),
        "rules_kept_positive": rule_stats.get("rules_kept_positive"),
        "rules_kept_negative": rule_stats.get("rules_kept_negative"),
        "llm_first_order_rule_count": rule_stats.get("llm_first_order_rule_count"),
        "llm_multi_antecedent_rule_count": rule_stats.get("llm_multi_antecedent_rule_count"),
        "high_order_rule_count": rule_stats.get("high_order_rule_count"),
        "path_rule_count": rule_stats.get("path_rule_count"),
        "treatment_pair_count": rule_stats.get("treatment_pair_count"),
        "herb_pair_count": rule_stats.get("herb_pair_count"),
    }


def _resolve_rank_k(configured_k: int, labels: np.ndarray) -> int:
    if configured_k is not None and configured_k > 0:
        return configured_k
    avg_count = int(round(float(np.mean(np.sum(labels, axis=1)))))
    return max(avg_count, 1)


def compute_metrics_from_arrays(model: nn.Module, args: argparse.Namespace, preds: dict[str, np.ndarray], labels: dict[str, np.ndarray]):
    treatment_rank_k = _resolve_rank_k(args.treatment_top_k, labels["treatment"])
    herb_rank_k = _resolve_rank_k(args.herb_top_k, labels["herb"])
    return {
        "diag": compute_multiclass_metrics(preds["diag"], labels["diag"], model.num_tcm_diag, top_k=args.diag_top_k),
        "syndrome": compute_multiclass_metrics(
            preds["syndrome"], labels["syndrome"], model.num_syndrome, top_k=args.syndrome_top_k
        ),
        "treatment": compute_multilabel_metrics(
            preds["treatment"],
            labels["treatment"],
            threshold=args.multilabel_threshold,
            rank_k=treatment_rank_k,
        ),
        "herb": compute_multilabel_metrics(
            preds["herb"],
            labels["herb"],
            threshold=args.multilabel_threshold,
            rank_k=herb_rank_k,
        ),
    }


def _compute_multiclass_weights(series: pd.Series, label_map: dict[int, int], max_weight: float) -> torch.Tensor:
    counts = torch.zeros(len(label_map), dtype=torch.float32)
    for label in series.dropna():
        mapped = label_map.get(int(label))
        if mapped is not None:
            counts[mapped] += 1
    nonzero = counts > 0
    weights = torch.ones_like(counts)
    if nonzero.any():
        weights[nonzero] = counts[nonzero].sum() / (nonzero.sum().float() * counts[nonzero])
    return weights.clamp(max=max_weight)


def _compute_multilabel_pos_weight(series: pd.Series, label_map: dict[int, int], parser, max_weight: float) -> torch.Tensor:
    pos_counts = torch.zeros(len(label_map), dtype=torch.float32)
    for labels in series.dropna().apply(parser):
        for label in labels:
            mapped = label_map.get(int(label))
            if mapped is not None:
                pos_counts[mapped] += 1
    total = float(len(series))
    neg_counts = total - pos_counts
    return (neg_counts / pos_counts.clamp(min=1.0)).clamp(min=1.0, max=max_weight)


def attach_training_statistics(args: argparse.Namespace, train_df: pd.DataFrame, mappings: dict[str, Any]) -> None:
    age_std = float(train_df[args.age_col].std())
    args.age_mean = float(train_df[args.age_col].mean())
    args.age_std = age_std if age_std > 1e-6 else 1.0

    if args.disable_loss_weights:
        args.diag_class_weights = None
        args.syndrome_class_weights = None
        args.treatment_pos_weight = None
        args.herb_pos_weight = None
        return

    args.diag_class_weights = _compute_multiclass_weights(
        train_df[args.tcm_diag_col], mappings["tcm_diag_map"], args.max_class_weight
    )
    args.syndrome_class_weights = _compute_multiclass_weights(
        train_df[args.syndrome_col], mappings["syndrome_map"], args.max_class_weight
    )
    args.treatment_pos_weight = _compute_multilabel_pos_weight(
        train_df[args.treatment_col], mappings["treatment_map"], parse_list_field, args.max_pos_weight
    )
    args.herb_pos_weight = _compute_multilabel_pos_weight(
        train_df[args.herb_col], mappings["herb_map"], parse_herb_dict_keys, args.max_pos_weight
    )


def make_loss_functions(args: argparse.Namespace, device: torch.device):
    diag_weight = args.diag_class_weights.to(device) if args.diag_class_weights is not None else None
    syndrome_weight = args.syndrome_class_weights.to(device) if args.syndrome_class_weights is not None else None
    treatment_pos_weight = args.treatment_pos_weight.to(device) if args.treatment_pos_weight is not None else None
    herb_pos_weight = args.herb_pos_weight.to(device) if args.herb_pos_weight is not None else None
    return (
        nn.CrossEntropyLoss(weight=diag_weight),
        nn.CrossEntropyLoss(weight=syndrome_weight),
        nn.BCEWithLogitsLoss(pos_weight=treatment_pos_weight),
        nn.BCEWithLogitsLoss(pos_weight=herb_pos_weight),
    )


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def compute_task_loss(logits: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], loss_fns, args: argparse.Namespace) -> torch.Tensor:
    loss_diag, loss_syndrome, loss_treatment, loss_herb = loss_fns
    return (
        args.lambda_diag * loss_diag(logits["diag"], batch["tcm_diag"])
        + args.lambda_syndrome * loss_syndrome(logits["syndrome"], batch["syndrome"])
        + args.lambda_treatment * loss_treatment(logits["treatment"], batch["treatment"])
        + args.lambda_herb * loss_herb(logits["herb"], batch["herb"])
    )


def compute_rule_consistency_loss(outputs: dict[str, Any], args: argparse.Namespace) -> torch.Tensor:
    selected_tasks = {
        task.strip()
        for task in str(args.rule_consistency_tasks).split(",")
        if task.strip()
    }
    if not selected_tasks:
        return next(iter(outputs["fused"].values())).new_tensor(0.0)

    total = None
    count = 0
    threshold = args.rule_consistency_threshold
    detach_symbolic = not bool(args.rule_consistency_no_detach)
    for task in TASKS:
        if task not in selected_tasks:
            continue
        symbolic_logits = outputs[args.rule_consistency_source][task]
        if detach_symbolic:
            symbolic_logits = symbolic_logits.detach()
        fused_logits = outputs["fused"][task]
        if task in MULTILABEL_TASKS:
            symbolic_prob = torch.sigmoid(symbolic_logits)
            fused_prob = torch.sigmoid(fused_logits)
            mask = symbolic_prob >= threshold
            if bool(mask.any()):
                task_loss = (torch.relu(symbolic_prob - fused_prob).pow(2) * mask.float()).sum() / mask.float().sum().clamp(min=1.0)
            else:
                continue
        else:
            symbolic_prob = torch.softmax(symbolic_logits, dim=1)
            fused_prob = torch.softmax(fused_logits, dim=1)
            top_prob, top_idx = symbolic_prob.max(dim=1)
            mask = top_prob >= threshold
            if bool(mask.any()):
                fused_top = fused_prob.gather(1, top_idx.unsqueeze(1)).squeeze(1)
                task_loss = torch.relu(top_prob - fused_top).pow(2)[mask].mean()
            else:
                continue
        total = task_loss if total is None else total + task_loss
        count += 1
    if total is None:
        return next(iter(outputs["fused"].values())).new_tensor(0.0)
    return total / max(count, 1)


def rule_beta_for_epoch(args: argparse.Namespace, epoch: int) -> float:
    target = float(args.beta_rule)
    warmup = int(args.rule_warmup_epochs)
    if warmup <= 0:
        return target
    if epoch <= 1:
        return 0.0
    return target * min(1.0, float(epoch - 1) / float(warmup))


def compute_total_loss(
    outputs: dict[str, Any],
    batch: dict[str, torch.Tensor],
    loss_fns,
    args: argparse.Namespace,
    beta_rule: float,
):
    fused_loss = compute_task_loss(outputs["fused"], batch, loss_fns, args)
    neural_loss = compute_task_loss(outputs["neural"], batch, loss_fns, args)
    symbolic_loss = compute_task_loss(outputs["symbolic"], batch, loss_fns, args)
    rule_loss = compute_rule_consistency_loss(outputs, args)
    total = (
        fused_loss
        + args.alpha_neural * neural_loss
        + args.alpha_symbolic * symbolic_loss
        + beta_rule * rule_loss
    )
    return total, {
        "loss_total": total.detach(),
        "loss_fused": fused_loss.detach(),
        "loss_neural": neural_loss.detach(),
        "loss_symbolic": symbolic_loss.detach(),
        "loss_rule": rule_loss.detach(),
    }


def summarize_tensor_list(values: list[torch.Tensor]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0}
    arr = torch.cat([v.reshape(-1).detach().cpu() for v in values]).numpy()
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
    }


def summarize_source_weight_list(values: list[torch.Tensor]) -> dict[str, float]:
    if not values:
        return {"first": 0.0, "high": 0.0, "path": 0.0, "co": 0.0}
    arr = torch.cat([v.detach().cpu().float() for v in values], dim=0)
    names = ("first", "high", "path", "co")
    return {name: float(arr[:, i].mean().item()) for i, name in enumerate(names[: arr.shape[1]])}


def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, loss_fns, args, epoch):
    model.train()
    totals = {
        "loss_total": 0.0,
        "loss_fused": 0.0,
        "loss_neural": 0.0,
        "loss_symbolic": 0.0,
        "loss_rule": 0.0,
    }
    beta_rule = rule_beta_for_epoch(args, epoch)
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(dataloader, desc=f"Training epoch {epoch}", leave=False)

    def optimizer_step_with_optional_scheduler():
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()
        optimizer_was_stepped = (not scaler.is_enabled()) or scale_after >= scale_before
        if scheduler is not None and optimizer_was_stepped:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(progress):
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast("cuda", enabled=args.use_amp and device.type == "cuda"):
            outputs = model(batch, mode="train")
            loss, parts = compute_total_loss(outputs, batch, loss_fns, args, beta_rule)
            loss = loss / args.gradient_accumulation_steps

        scaler.scale(loss).backward()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer_step_with_optional_scheduler()

        for key in totals:
            totals[key] += float(parts[key].item())
        progress.set_postfix(loss=totals["loss_total"] / (step + 1), beta_rule=beta_rule)

    if len(dataloader) % args.gradient_accumulation_steps != 0:
        optimizer_step_with_optional_scheduler()

    return {key: value / max(len(dataloader), 1) for key, value in totals.items()}


def print_epoch_summary(
    model_name: str,
    epoch: int,
    args: argparse.Namespace,
    train_parts: dict[str, float],
    oracle_loss: float,
    oracle_metrics: dict[str, dict[str, float]],
    predicted_loss: float,
    predicted_metrics: dict[str, dict[str, float]],
) -> None:
    print(f"[{model_name}]")
    print(
        f"Epoch {epoch}/{args.epochs}: "
        f"train_loss={train_parts['loss_total']:.4f}, "
        f"fused={train_parts['loss_fused']:.4f}, "
        f"rule={train_parts['loss_rule']:.4f}, "
        f"valid_oracle_loss={oracle_loss:.4f}, "
        f"valid_predicted_loss={predicted_loss:.4f}, "
        f"valid_predicted_chain_score={compute_chain_score(predicted_metrics):.4f}"
    )
    for name, metrics in [("Oracle", oracle_metrics), ("Predicted", predicted_metrics)]:
        print(
            f"  {name}: "
            f"diag={metrics['diag']['score']:.4f}, "
            f"syndrome={metrics['syndrome']['score']:.4f}, "
            f"treatment={metrics['treatment']['score']:.4f}, "
            f"herb={metrics['herb']['score']:.4f}, "
            f"chain={compute_chain_score(metrics):.4f}"
        )


def evaluate_model(model, dataloader, device, loss_fns, args, mode="predict", collect_sources=False):
    model.eval()
    total_loss = 0.0
    sources = ["fused", "neural", "symbolic"] if collect_sources else ["fused"]
    all_preds = {source: {task: [] for task in TASKS} for source in sources}
    all_labels = {task: [] for task in TASKS}
    gate_values = {task: [] for task in TASKS}
    disagreement_values = {task: [] for task in TASKS}
    rule_source_weight_values = {task: [] for task in TASKS}

    with torch.no_grad():
        progress = tqdm(dataloader, desc=f"Evaluating {mode}", leave=False)
        for batch in progress:
            batch = move_batch_to_device(batch, device)
            with torch.amp.autocast("cuda", enabled=args.use_amp and device.type == "cuda"):
                outputs = model(batch, mode=mode)
                loss = compute_task_loss(outputs["fused"], batch, loss_fns, args)
            total_loss += loss.item()

            for source in sources:
                for task in TASKS:
                    all_preds[source][task].append(outputs[source][task].detach().cpu().numpy())
            for task in TASKS:
                all_labels[task].append(batch[LABEL_KEYS[task]].detach().cpu().numpy())
                gate_values[task].append(outputs["gates"][task].detach().cpu())
                disagreement_values[task].append(outputs["disagreements"][task].detach().cpu())
                details = outputs.get("rule_details", {}).get(task, {})
                source_weights = details.get("source_weights") if isinstance(details, dict) else None
                if torch.is_tensor(source_weights):
                    rule_source_weight_values[task].append(source_weights.detach().cpu())

    labels = {task: np.concatenate(values, axis=0) for task, values in all_labels.items()}
    source_metrics = {}
    for source in sources:
        preds = {task: np.concatenate(all_preds[source][task], axis=0) for task in TASKS}
        source_metrics[source] = compute_metrics_from_arrays(model, args, preds, labels)

    diagnostics = {
        "source_metrics": source_metrics,
        "gate_stats": {task: summarize_tensor_list(gate_values[task]) for task in TASKS},
        "disagreement_stats": {task: summarize_tensor_list(disagreement_values[task]) for task in TASKS},
        "rule_source_weight_stats": {
            task: summarize_source_weight_list(rule_source_weight_values[task]) for task in TASKS
        },
    }
    return total_loss / max(len(dataloader), 1), source_metrics["fused"], diagnostics


def configure_optimizer(model: nn.Module, args: argparse.Namespace):
    groups = {"bert": [], "neural_head": [], "rule": [], "fusion": []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("neural.bert."):
            groups["bert"].append(param)
        elif name.startswith("neural."):
            groups["neural_head"].append(param)
        elif name.startswith("reasoner."):
            groups["rule"].append(param)
        else:
            groups["fusion"].append(param)

    optimizer_groups = []
    if groups["bert"]:
        optimizer_groups.append({"params": groups["bert"], "lr": args.lr, "name": "bert"})
    if groups["neural_head"]:
        optimizer_groups.append({"params": groups["neural_head"], "lr": args.head_lr, "name": "neural_head"})
    if groups["rule"]:
        optimizer_groups.append({"params": groups["rule"], "lr": args.rule_lr, "name": "rule"})
    if groups["fusion"]:
        optimizer_groups.append({"params": groups["fusion"], "lr": args.fusion_lr, "name": "fusion"})
    print(
        "Optimizer parameter groups: "
        + ", ".join(f"{group['name']}={len(group['params'])}" for group in optimizer_groups)
    )
    return optimizer_groups


def load_optional_neural_checkpoint(model: nn.Module, args: argparse.Namespace, device: torch.device) -> None:
    if not args.neural_checkpoint:
        return
    state = torch.load(args.neural_checkpoint, map_location=device)
    current = model.neural.state_dict()
    compatible = {k: v for k, v in state.items() if k in current and tuple(current[k].shape) == tuple(v.shape)}
    missing, unexpected = model.neural.load_state_dict(compatible, strict=False)
    print(
        f"Loaded neural checkpoint {args.neural_checkpoint}: "
        f"compatible={len(compatible)}, missing={len(missing)}, unexpected={len(unexpected)}"
    )


def run_single_model(
    model_name: str,
    args: argparse.Namespace,
    mappings: dict[str, Any],
    token_map: dict[int, int],
    rule_index,
    train_loader,
    valid_loader,
    test_loader,
    device: torch.device,
    model_factory: Callable[..., nn.Module] | None = None,
    variant_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_dir = Path(args.output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    factory = model_factory or create_dinsr_model
    model = factory(args, mappings, token_map, rule_index, model_name).to(device)
    load_optional_neural_checkpoint(model, args, device)
    loss_fns = make_loss_functions(args, device)

    optimizer_groups = configure_optimizer(model, args)
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=args.weight_decay)
    total_steps = max(1, math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.epochs)
    try:
        from transformers import get_linear_schedule_with_warmup
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Training requires `transformers`. Please install it or use the existing project environment."
        ) from exc

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp and device.type == "cuda")

    config_payload = {
        "model_name": model_name,
        "variant": variant_metadata or MODEL_VARIANTS.get(model_name, model_name),
        "args": vars(args),
        "protocol": protocol_dict(args, rule_index.rule_stats),
        "rule_stats": rule_index.rule_stats,
    }
    save_json(model_dir / "config.json", config_payload)

    best_score = -float("inf")
    best_path = model_dir / "best_model.pt"
    epochs_without_improvement = 0
    train_log = []

    for epoch in range(1, args.epochs + 1):
        train_parts = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, loss_fns, args, epoch)
        valid_oracle_loss, valid_oracle_metrics, valid_oracle_diag = evaluate_model(
            model, valid_loader, device, loss_fns, args, mode="oracle"
        )
        valid_predicted_loss, valid_predicted_metrics, valid_predicted_diag = evaluate_model(
            model, valid_loader, device, loss_fns, args, mode="predict"
        )
        valid_score = compute_chain_score(valid_predicted_metrics)
        row = {
            "epoch": epoch,
            **train_parts,
            "beta_rule_effective": rule_beta_for_epoch(args, epoch),
            "valid_oracle_loss": valid_oracle_loss,
            "valid_predicted_loss": valid_predicted_loss,
            **flatten_metrics("valid_oracle", valid_oracle_metrics),
            **flatten_metrics("valid_predicted", valid_predicted_metrics),
        }
        for task, stats in valid_predicted_diag["gate_stats"].items():
            row[f"valid_predicted_gate_{task}_mean"] = stats["mean"]
        for task, stats in valid_predicted_diag["disagreement_stats"].items():
            row[f"valid_predicted_disagreement_{task}_mean"] = stats["mean"]
        for task, stats in valid_predicted_diag.get("rule_source_weight_stats", {}).items():
            row[f"valid_predicted_rule_source_{task}"] = json.dumps(stats, ensure_ascii=False)
            for source, value in stats.items():
                row[f"valid_predicted_source_weight_{task}_{source}"] = value
        train_log.append(row)
        train_log_df = pd.DataFrame(train_log)
        train_log_df.to_csv(model_dir / "training_log.csv", index=False, encoding="utf-8-sig")

        print_epoch_summary(
            model_name,
            epoch,
            args,
            train_parts,
            valid_oracle_loss,
            valid_oracle_metrics,
            valid_predicted_loss,
            valid_predicted_metrics,
        )
        if args.enable_multi_granularity_rules:
            source_parts = []
            for task, stats in valid_predicted_diag.get("rule_source_weight_stats", {}).items():
                source_parts.append(
                    f"{task}=first:{stats.get('first', 0.0):.3f}/"
                    f"high:{stats.get('high', 0.0):.3f}/"
                    f"path:{stats.get('path', 0.0):.3f}/"
                    f"co:{stats.get('co', 0.0):.3f}"
                )
            if source_parts:
                print("  Rule source weights: " + "; ".join(source_parts))

        if valid_score > best_score + args.min_delta:
            best_score = valid_score
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_without_improvement += 1

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(f"[{model_name}] early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(best_path, map_location=device))
    valid_oracle_loss, valid_oracle_metrics, valid_oracle_diag = evaluate_model(
        model, valid_loader, device, loss_fns, args, mode="oracle", collect_sources=True
    )
    valid_predicted_loss, valid_predicted_metrics, valid_predicted_diag = evaluate_model(
        model, valid_loader, device, loss_fns, args, mode="predict", collect_sources=True
    )
    test_oracle_loss, test_oracle_metrics, test_oracle_diag = evaluate_model(
        model, test_loader, device, loss_fns, args, mode="oracle", collect_sources=True
    )
    test_predicted_loss, test_predicted_metrics, test_predicted_diag = evaluate_model(
        model, test_loader, device, loss_fns, args, mode="predict", collect_sources=True
    )

    protocol = protocol_dict(args, rule_index.rule_stats)
    valid_payload = {
        "protocol": protocol,
        "summary": {
            "oracle_chain_score": compute_chain_score(valid_oracle_metrics),
            "predicted_chain_score": compute_chain_score(valid_predicted_metrics),
        },
        "oracle": {
            "loss": valid_oracle_loss,
            "metrics": valid_oracle_metrics,
            "diagnostics": valid_oracle_diag,
        },
        "predicted": {
            "loss": valid_predicted_loss,
            "metrics": valid_predicted_metrics,
            "diagnostics": valid_predicted_diag,
        },
    }
    test_payload = {
        "protocol": protocol,
        "summary": {
            "oracle_chain_score": compute_chain_score(test_oracle_metrics),
            "predicted_chain_score": compute_chain_score(test_predicted_metrics),
        },
        "oracle": {
            "loss": test_oracle_loss,
            "metrics": test_oracle_metrics,
            "diagnostics": test_oracle_diag,
        },
        "predicted": {
            "loss": test_predicted_loss,
            "metrics": test_predicted_metrics,
            "diagnostics": test_predicted_diag,
        },
    }
    save_json(model_dir / "valid_metrics.json", valid_payload)
    save_json(model_dir / "test_metrics.json", test_payload)
    gate_payload = {
        "oracle": {
            "gate_stats": test_oracle_diag["gate_stats"],
            "disagreement_stats": test_oracle_diag["disagreement_stats"],
            "rule_source_weight_stats": test_oracle_diag["rule_source_weight_stats"],
        },
        "predicted": {
            "gate_stats": test_predicted_diag["gate_stats"],
            "disagreement_stats": test_predicted_diag["disagreement_stats"],
            "rule_source_weight_stats": test_predicted_diag["rule_source_weight_stats"],
        },
    }
    save_json(model_dir / "gate_statistics.json", gate_payload)

    result = {
        "model_name": model_name,
        "status": "ok",
        "valid_loss": valid_predicted_loss,
        "test_loss": test_predicted_loss,
        **flatten_metrics("valid_oracle", valid_oracle_metrics),
        **flatten_metrics("valid_predicted", valid_predicted_metrics),
        **flatten_metrics("test_oracle", test_oracle_metrics),
        **flatten_metrics("test_predicted", test_predicted_metrics),
        "mean_gate_diag": test_predicted_diag["gate_stats"]["diag"]["mean"],
        "mean_gate_syndrome": test_predicted_diag["gate_stats"]["syndrome"]["mean"],
        "mean_gate_treatment": test_predicted_diag["gate_stats"]["treatment"]["mean"],
        "mean_gate_herb": test_predicted_diag["gate_stats"]["herb"]["mean"],
        "llm_first_order_rule_count": rule_index.rule_stats.get("llm_first_order_rule_count"),
        "llm_multi_antecedent_rule_count": rule_index.rule_stats.get("llm_multi_antecedent_rule_count"),
        "high_order_rule_count": rule_index.rule_stats.get("high_order_rule_count"),
        "path_rule_count": rule_index.rule_stats.get("path_rule_count"),
        "treatment_pair_count": rule_index.rule_stats.get("treatment_pair_count"),
        "herb_pair_count": rule_index.rule_stats.get("herb_pair_count"),
    }
    for task, stats in test_predicted_diag.get("rule_source_weight_stats", {}).items():
        result[f"by_source_weight_{task}"] = json.dumps(stats, ensure_ascii=False)
        for source, value in stats.items():
            result[f"by_source_weight_{task}_{source}"] = value
    return result


def main(
    args: argparse.Namespace,
    model_name: str = "DiNSR",
    model_factory: Callable[..., nn.Module] | None = None,
    rule_index_transform: Callable[[Any, argparse.Namespace], Any] | None = None,
    variant_metadata: dict[str, Any] | None = None,
) -> None:
    if args.local_files_only:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    elif args.hf_mirror:
        os.environ["HF_ENDPOINT"] = args.hf_mirror

    set_seed(args.seed)
    print_gpu_info()
    device = get_device(args.gpu_id)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data: {args.data_path}")
    if not Path(args.data_path).is_file():
        raise FileNotFoundError(
            "Clinical benchmark data was not found. The public release does not ship "
            "`data/tcm_benchmark.csv`; authorized users should generate it with "
            "`python prepare_release_data.py` or pass an approved local file via "
            "`--data_path`."
        )
    df = read_csv_with_encoding_fallback(args.data_path)
    mappings = build_label_mappings(
        df,
        args.tcm_diag_col,
        args.syndrome_col,
        args.treatment_col,
        args.herb_col,
        args.western_diag_col,
        mapping_mode=args.label_mapping_mode,
    )
    token_map = build_token_map(df, args.token_col)
    print(
        "Label sizes: "
        f"tcm_diag={len(mappings['tcm_diag_map'])}, "
        f"syndrome={len(mappings['syndrome_map'])}, "
        f"treatment={len(mappings['treatment_map'])}, "
        f"herb={len(mappings['herb_map'])}, "
        f"token={len(token_map)}"
    )
    save_json(output_dir / "label_mappings.json", mappings)
    save_json(output_dir / "token_map.json", token_map)

    if args.enable_multi_granularity_rules:
        print(f"Building multi-granularity rule index: {args.llm_rule_path}")
        rule_index = build_multi_granularity_rule_index(
            args.llm_rule_path,
            mappings=mappings,
            token_map=token_map,
            max_rules=args.max_llm_rules,
            high_order_rule_path=args.high_order_rule_path,
            path_rule_path=args.path_rule_path,
            treatment_pair_rule_path=args.treatment_pair_rule_path,
            herb_pair_rule_path=args.herb_pair_rule_path,
        )
    else:
        print(f"Building LLM-audited rule index: {args.llm_rule_path}")
        rule_index = build_llm_audited_rule_index(
            args.llm_rule_path,
            mappings=mappings,
            token_map=token_map,
            max_rules=args.max_llm_rules,
        )
    if rule_index_transform is not None:
        rule_index = rule_index_transform(rule_index, args)
    save_json(output_dir / "llm_rule_statistics.json", rule_index.rule_stats)
    print(f"Rule stats: {rule_index.rule_stats}")

    if "split" not in df.columns:
        raise ValueError("Benchmark data must contain the fixed split column.")
    train_df = df[df["split"] == "train"].copy()
    valid_df = df[df["split"] == "valid"].copy()
    test_df = df[df["split"] == "test"].copy()
    if min(len(train_df), len(valid_df), len(test_df)) == 0:
        raise ValueError("The fixed split column must contain train, valid, and test rows.")
    attach_training_statistics(args, train_df, mappings)
    print(f"Dataset split: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")
    print(f"Age normalization: mean={args.age_mean:.4f}, std={args.age_std:.4f}")

    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Training requires `transformers` for AutoTokenizer/AutoModel. "
            "Please install it or activate the project environment."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name, local_files_only=args.local_files_only)
    train_dataset = TCMNeuralSymbolicDataset(train_df, tokenizer, mappings, token_map, args)
    valid_dataset = TCMNeuralSymbolicDataset(valid_df, tokenizer, mappings, token_map, args)
    test_dataset = TCMNeuralSymbolicDataset(test_df, tokenizer, mappings, token_map, args)
    pin_memory = bool(args.pin_memory and device.type == "cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=neural_symbolic_collate_fn,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=neural_symbolic_collate_fn,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=neural_symbolic_collate_fn,
        pin_memory=pin_memory,
    )

    print(f"\n========== Running {model_name} ==========")
    result = run_single_model(
        model_name,
        args,
        mappings,
        token_map,
        rule_index,
        train_loader,
        valid_loader,
        test_loader,
        device,
        model_factory=model_factory,
        variant_metadata=variant_metadata,
    )
    report = (
        f"{model_name} benchmark result\n"
        f"validation predicted-chain score: {result['valid_predicted_chain_score']:.6f}\n"
        f"test predicted-chain score: {result['test_predicted_chain_score']:.6f}\n"
        f"diagnosis macro-F1: {result['test_predicted_diag_score']:.6f}\n"
        f"syndrome macro-F1: {result['test_predicted_syndrome_score']:.6f}\n"
        f"treatment micro-F1: {result['test_predicted_treatment_score']:.6f}\n"
        f"herb micro-F1: {result['test_predicted_herb_score']:.6f}\n"
    )
    with open(output_dir / "final_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nFinished. Results saved to {output_dir}")
    print(report)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the DiNSR TCM benchmark model.")
    parser.add_argument("--data_path", type=str, default=str(PROJECT_ROOT / "data" / "tcm_benchmark.csv"))
    parser.add_argument("--llm_rule_path", type=str, default=str(PROJECT_ROOT / "rules" / "llm_audited_rules.csv"))
    parser.add_argument(
        "--high_order_rule_path",
        type=str,
        default=str(PROJECT_ROOT / "rules" / "high_order_rules.csv"),
    )
    parser.add_argument(
        "--path_rule_path",
        type=str,
        default=str(PROJECT_ROOT / "rules" / "path_rules.csv"),
    )
    parser.add_argument(
        "--treatment_pair_rule_path",
        type=str,
        default=str(PROJECT_ROOT / "rules" / "treatment_pair_rules.csv"),
    )
    parser.add_argument(
        "--herb_pair_rule_path",
        type=str,
        default=str(PROJECT_ROOT / "rules" / "herb_pair_rules.csv"),
    )
    parser.add_argument("--no-high_order_rule_path", dest="high_order_rule_path", action="store_const", const="")
    parser.add_argument("--no-path_rule_path", dest="path_rule_path", action="store_const", const="")
    parser.add_argument("--no-treatment_pair_rule_path", dest="treatment_pair_rule_path", action="store_const", const="")
    parser.add_argument("--no-herb_pair_rule_path", dest="herb_pair_rule_path", action="store_const", const="")
    parser.add_argument("--enable_multi_granularity_rules", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rule_source_attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rule_source_attention_hidden_dim", type=int, default=32)
    parser.add_argument("--high_order_activation", choices=["soft_and", "mean"], default="soft_and")
    parser.add_argument("--cooccurrence_weight_scale", type=float, default=0.5)
    parser.add_argument("--detach_cooccurrence_probs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "dinsr"))
    parser.add_argument("--bert_model_name", type=str, default="hfl/chinese-macbert-base")
    parser.add_argument("--hf_mirror", type=str, default="")
    parser.add_argument("--local_files_only", action="store_true")

    parser.add_argument("--label_mapping_mode", choices=["original", "compact"], default="original")
    parser.add_argument("--sex_col", type=str, default="sex")
    parser.add_argument("--age_col", type=str, default="age")
    parser.add_argument("--western_diag_col", type=str, default="initial_diagnosis")
    parser.add_argument("--tcm_diag_col", type=str, default="tcm_diagnosis")
    parser.add_argument("--syndrome_col", type=str, default="syndrome")
    parser.add_argument("--treatment_col", type=str, default="treatment_principles")
    parser.add_argument("--herb_col", type=str, default="herbs")
    parser.add_argument("--text_col", type=str, default="medical_record_summary")
    parser.add_argument("--token_col", type=str, default="medical_text_lexicon")

    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rule_dropout", type=float, default=0.10)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--freeze_bert", action="store_true")

    parser.add_argument("--max_llm_rules", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", "--learning_rate", dest="lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", "--head_learning_rate", dest="head_lr", type=float, default=1e-3)
    parser.add_argument("--rule_lr", "--symbolic_learning_rate", dest="rule_lr", type=float, default=5e-4)
    parser.add_argument("--fusion_lr", "--fusion_learning_rate", dest="fusion_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", "--random_seed", dest="seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=2.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--multilabel_threshold", type=float, default=0.5)
    parser.add_argument("--diag_top_k", type=int, default=3)
    parser.add_argument("--syndrome_top_k", type=int, default=3)
    parser.add_argument("--treatment_top_k", type=int, default=5)
    parser.add_argument("--herb_top_k", type=int, default=13)
    parser.add_argument("--chain_inference", choices=["soft", "hard"], default="soft")
    parser.add_argument("--max_class_weight", type=float, default=10.0)
    parser.add_argument("--max_pos_weight", type=float, default=20.0)
    parser.add_argument("--disable_loss_weights", action="store_true")

    parser.add_argument("--lambda_diag", type=float, default=1.0)
    parser.add_argument("--lambda_syndrome", type=float, default=1.0)
    parser.add_argument("--lambda_treatment", type=float, default=1.0)
    parser.add_argument("--lambda_herb", type=float, default=1.0)
    parser.add_argument("--alpha_neural", type=float, default=0.2)
    parser.add_argument("--alpha_symbolic", type=float, default=0.1)
    parser.add_argument("--beta_rule", type=float, default=0.02)
    parser.add_argument("--rule_warmup_epochs", type=int, default=2)
    parser.add_argument("--rule_consistency_tasks", type=str, default="treatment,herb")
    parser.add_argument("--rule_consistency_source", choices=["symbolic", "symbolic_pos"], default="symbolic_pos")
    parser.add_argument("--rule_consistency_no_detach", action="store_true")
    parser.add_argument("--rule_consistency_threshold", type=float, default=0.65)

    parser.add_argument(
        "--fusion_gate_hidden_dim",
        type=int,
        default=32,
    )
    parser.add_argument("--fusion_gamma_diag", type=float, default=0.05)
    parser.add_argument("--fusion_gamma_syndrome", type=float, default=0.10)
    parser.add_argument("--fusion_gamma_treatment", type=float, default=0.35)
    parser.add_argument("--fusion_gamma_herb", type=float, default=0.35)

    parser.add_argument("--neural_checkpoint", type=str, default="")
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
