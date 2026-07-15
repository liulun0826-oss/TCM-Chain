from __future__ import annotations

import argparse
import math
import pickle
import threading
import time
import traceback
import warnings
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from .models import BASELINE_SPECS, BaselineSpec, build_torch_model
from .utils import (
    TCMComparisonDataset,
    attach_training_statistics,
    build_train_cooccurrence_graph,
    comparison_collate_fn,
    compute_chain_score,
    compute_metrics_from_arrays,
    dataframe_label_arrays,
    evaluate_torch_chain,
    flatten_metrics,
    make_loss_functions,
    print_epoch_summary,
    print_metrics_table,
    print_score_lines,
    resolve_rank_k,
    save_json,
    train_torch_one_epoch,
)
from dataset import build_label_mappings
from metrics import compute_multiclass_metrics, compute_multilabel_metrics
from symbolic_dataset import build_token_map, parse_token_field
from utils import get_device, print_gpu_info, read_csv_with_encoding_fallback, set_seed


TEXT_TORCH_MODELS = {"TextCNNChain", "BiLSTMAttentionChain", "MacBERTChain", "MMoEChain", "PLEChain"}
MODEL_SET_RUNNERS = {
    "neural": ("torch", "torch_graph"),
    "all": ("torch", "torch_graph", "bge"),
}
MODEL_SET_CHOICES = tuple(MODEL_SET_RUNNERS)


def _training_regime(spec: BaselineSpec) -> str:
    if spec.runner in {"torch", "torch_graph"}:
        return "torch_minibatch"
    if spec.runner == "bge":
        return "bge_encode_then_sklearn_fit"
    raise ValueError(f"Unsupported runner {spec.runner} for {spec.name}")


def _runner_plan_label(runner: str, args: argparse.Namespace, device: torch.device) -> str:
    if runner in {"torch", "torch_graph"}:
        graph_suffix = " with train-split graphs" if runner == "torch_graph" else ""
        return f"Torch mini-batch on {device}{graph_suffix}; epochs={args.epochs}, batch_size={args.batch_size}"
    if runner == "bge":
        return f"BGE batch encoding on {device}, then sklearn CPU stagewise LR fit"
    raise ValueError(f"Unsupported runner: {runner}")


def _model_names_for_set(model_set: str) -> list[str]:
    runners = MODEL_SET_RUNNERS[model_set]
    return [name for name, spec in BASELINE_SPECS.items() if spec.runner in runners]


def _model_selection_source(args: argparse.Namespace) -> str:
    if args.model_list:
        return "explicit --model_list"
    return f"--model_set {args.model_set}"


def print_execution_plan(model_names: list[str], args: argparse.Namespace, device: torch.device) -> None:
    print(f"Model selection: {_model_selection_source(args)}")
    print(f"Selected models ({len(model_names)}): {', '.join(model_names)}")
    print("Execution plan:")
    for runner in ("torch", "torch_graph", "bge"):
        names = [name for name in model_names if BASELINE_SPECS[name].runner == runner]
        if names:
            print(f"  {_runner_plan_label(runner, args, device)}: {', '.join(names)}")


def protocol_dict(args: argparse.Namespace, spec: BaselineSpec) -> dict[str, Any]:
    return {
        "model": spec.name,
        "setting": spec.setting,
        "runner": spec.runner,
        "training_regime": _training_regime(spec),
        "paper": spec.paper,
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
        "split": "fixed split column when present; otherwise 75/12.5/12.5 with random_seed",
        "random_seed": args.random_seed,
        "rules_used": False,
        "cooccurrence_graph_source": "train split only" if spec.setting == "C" else None,
    }


def select_model_names(args: argparse.Namespace) -> list[str]:
    if args.model_list:
        selected = [name.strip() for name in args.model_list.split(",") if name.strip()]
    else:
        selected = _model_names_for_set(args.model_set)
    unknown = [name for name in selected if name not in BASELINE_SPECS]
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}. Valid names: {list(BASELINE_SPECS.keys())}")
    return selected


def _run_with_heartbeat(label: str, action: Any, args: argparse.Namespace) -> Any:
    interval = max(float(getattr(args, "progress_heartbeat_seconds", 30.0)), 0.0)
    started = time.perf_counter()
    stop = threading.Event()
    reporter = None
    print(f"{label}...")

    if interval > 0:
        def report() -> None:
            while not stop.wait(interval):
                elapsed = time.perf_counter() - started
                print(f"  still running: {label} ({elapsed:.0f}s elapsed)")

        reporter = threading.Thread(target=report, name="baseline-progress-heartbeat", daemon=True)
        reporter.start()

    try:
        result = action()
    except Exception:
        elapsed = time.perf_counter() - started
        print(f"{label} failed after {elapsed:.1f}s.")
        raise
    else:
        elapsed = time.perf_counter() - started
        print(f"{label} finished in {elapsed:.1f}s.")
        return result
    finally:
        stop.set()
        if reporter is not None:
            reporter.join(timeout=0.2)


def _safe_float(value: float | None) -> float | None:
    if value is None:
        return None
    if math.isnan(float(value)):
        return None
    return float(value)


def _age_bucket(value: Any) -> str:
    try:
        age = int(float(value))
    except (TypeError, ValueError):
        return "age_missing"
    return f"age_{min(max(age // 10, 0), 12)}0s"


def _lexicon_document(row: pd.Series, args: argparse.Namespace) -> str:
    token_text = " ".join(f"lex_{token}" for token in sorted(set(parse_token_field(row[args.token_col]))))
    return token_text or "lex_empty"


def bge_documents(
    df: pd.DataFrame,
    args: argparse.Namespace,
    progress_desc: str | None = None,
) -> list[str]:
    documents = []
    rows = df.iterrows()
    if progress_desc:
        rows = tqdm(rows, total=len(df), desc=progress_desc, leave=False)
    for _, row in rows:
        metadata = (
            f" { _age_bucket(row.get(args.age_col)) }"
            f" sex_{row.get(args.sex_col, 'missing')}"
            f" western_{row.get(args.western_diag_col, 'missing')}"
        )
        documents.append(f"{_lexicon_document(row, args)}{metadata}")
    return documents


def _one_hot(values: np.ndarray, width: int) -> sparse.csr_matrix:
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    rows = np.arange(len(values))
    valid = (values >= 0) & (values < width)
    return sparse.csr_matrix((np.ones(int(valid.sum())), (rows[valid], values[valid])), shape=(len(values), width))


def _as_sparse(values: np.ndarray | sparse.spmatrix) -> sparse.csr_matrix:
    return values.tocsr() if sparse.issparse(values) else sparse.csr_matrix(values)


def _stack_features(base: np.ndarray | sparse.spmatrix, *features: np.ndarray | sparse.spmatrix) -> sparse.csr_matrix:
    return sparse.hstack([_as_sparse(base), *(_as_sparse(feature) for feature in features)], format="csr")


def _sigmoid(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(np.clip(shifted, -30.0, 30.0))
    return exp_scores / np.maximum(exp_scores.sum(axis=1, keepdims=True), 1e-8)


def _scores_to_multiclass_features(scores: np.ndarray, args: argparse.Namespace) -> sparse.csr_matrix:
    if args.chain_inference == "hard":
        return _one_hot(np.argmax(scores, axis=1), scores.shape[1])
    return _as_sparse(_softmax(scores))


def _scores_to_multilabel_features(scores: np.ndarray, args: argparse.Namespace) -> sparse.csr_matrix:
    probs = _sigmoid(scores)
    if args.chain_inference == "hard":
        probs = (probs >= float(args.multilabel_threshold)).astype(np.float32)
    return _as_sparse(probs)


def _logistic_estimator(args: argparse.Namespace) -> LogisticRegression:
    return LogisticRegression(
        C=float(args.logistic_c),
        max_iter=int(args.sklearn_max_iter),
        solver="liblinear",
        class_weight="balanced" if args.sklearn_class_weight_balanced else None,
    )


def _one_vs_rest_logistic(args: argparse.Namespace) -> OneVsRestClassifier:
    return OneVsRestClassifier(_logistic_estimator(args), n_jobs=int(args.sklearn_n_jobs))


def _raw_scores(estimator: Any, features: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(estimator.decision_function(features), dtype=np.float32)


def _multiclass_scores(estimator: Any, features: sparse.csr_matrix, width: int) -> np.ndarray:
    raw = _raw_scores(estimator, features)
    classes = np.asarray(getattr(estimator, "classes_", np.arange(width)), dtype=np.int64)
    if raw.ndim == 1:
        raw = np.stack([-raw, raw], axis=1) if len(classes) == 2 else raw[:, None]
    if raw.shape[1] == width and np.array_equal(classes, np.arange(width)):
        return raw.astype(np.float32)
    full = np.full((features.shape[0], width), -30.0, dtype=np.float32)
    for raw_index, class_index in enumerate(classes[: raw.shape[1]]):
        if 0 <= int(class_index) < width:
            full[:, int(class_index)] = raw[:, raw_index]
    return full


def _multilabel_scores(estimator: Any, features: sparse.csr_matrix, width: int) -> np.ndarray:
    raw = _raw_scores(estimator, features)
    if raw.ndim == 1:
        raw = raw[:, None]
    if raw.shape[1] == width:
        return raw.astype(np.float32)
    full = np.full((features.shape[0], width), -30.0, dtype=np.float32)
    full[:, : min(width, raw.shape[1])] = raw[:, : min(width, raw.shape[1])]
    return full


def _print_stage_validation_metric(
    task: str,
    scores: np.ndarray,
    labels: np.ndarray,
    mappings: dict[str, Any],
    args: argparse.Namespace,
    split_name: str,
) -> None:
    if task == "diag":
        metrics = compute_multiclass_metrics(scores, labels, len(mappings["tcm_diag_map"]), top_k=int(args.diag_top_k))
        detail = f"macro_f1={metrics['score']:.4f}, accuracy={metrics['accuracy']:.4f}"
    elif task == "syndrome":
        metrics = compute_multiclass_metrics(scores, labels, len(mappings["syndrome_map"]), top_k=int(args.syndrome_top_k))
        detail = f"macro_f1={metrics['score']:.4f}, accuracy={metrics['accuracy']:.4f}"
    else:
        rank_k = resolve_rank_k(args.treatment_top_k if task == "treatment" else args.herb_top_k, labels)
        metrics = compute_multilabel_metrics(
            scores,
            labels,
            threshold=float(args.multilabel_threshold),
            rank_k=rank_k,
        )
        detail = f"micro_f1={metrics['score']:.4f}, top{rank_k}_micro_f1={metrics['topk_micro_f1']:.4f}"
    print(f"  {split_name} predicted {task}: {detail}")


class LogisticRegressionChain:
    def __init__(self, mappings: dict[str, Any], args: argparse.Namespace):
        self.mappings = mappings
        self.args = args
        self.diag = _one_vs_rest_logistic(args)
        self.syndrome = _one_vs_rest_logistic(args)
        self.treatment = _one_vs_rest_logistic(args)
        self.herb = _one_vs_rest_logistic(args)

    @property
    def num_diag(self) -> int:
        return len(self.mappings["tcm_diag_map"])

    @property
    def num_syndrome(self) -> int:
        return len(self.mappings["syndrome_map"])

    def fit(
        self,
        base_features: sparse.csr_matrix,
        labels: dict[str, np.ndarray],
        monitor_features: sparse.csr_matrix | None = None,
        monitor_labels: dict[str, np.ndarray] | None = None,
        monitor_split: str = "valid",
    ) -> "LogisticRegressionChain":
        diag_true = _one_hot(labels["diag"], self.num_diag)
        syndrome_true = _one_hot(labels["syndrome"], self.num_syndrome)
        treatment_true = _as_sparse(labels["treatment"])
        _run_with_heartbeat("Sklearn stage 1/4: fitting diag", lambda: self.diag.fit(base_features, labels["diag"]), self.args)
        monitor_diag_features = None
        monitor_syndrome_features = None
        monitor_treatment_features = None
        if monitor_features is not None and monitor_labels is not None:
            diag_scores = _multiclass_scores(self.diag, monitor_features, self.num_diag)
            _print_stage_validation_metric("diag", diag_scores, monitor_labels["diag"], self.mappings, self.args, monitor_split)
            monitor_diag_features = _scores_to_multiclass_features(diag_scores, self.args)

        syndrome_features = _stack_features(base_features, diag_true)
        _run_with_heartbeat(
            "Sklearn stage 2/4: fitting syndrome",
            lambda: self.syndrome.fit(syndrome_features, labels["syndrome"]),
            self.args,
        )
        if monitor_features is not None and monitor_labels is not None and monitor_diag_features is not None:
            monitor_syndrome_input = _stack_features(monitor_features, monitor_diag_features)
            syndrome_scores = _multiclass_scores(self.syndrome, monitor_syndrome_input, self.num_syndrome)
            _print_stage_validation_metric(
                "syndrome",
                syndrome_scores,
                monitor_labels["syndrome"],
                self.mappings,
                self.args,
                monitor_split,
            )
            monitor_syndrome_features = _scores_to_multiclass_features(syndrome_scores, self.args)

        treatment_features = _stack_features(base_features, diag_true, syndrome_true)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"Label .* is present in all training examples\.")
            _run_with_heartbeat(
                "Sklearn stage 3/4: fitting treatment",
                lambda: self.treatment.fit(treatment_features, labels["treatment"]),
                self.args,
            )
            if (
                monitor_features is not None
                and monitor_labels is not None
                and monitor_diag_features is not None
                and monitor_syndrome_features is not None
            ):
                monitor_treatment_input = _stack_features(monitor_features, monitor_diag_features, monitor_syndrome_features)
                treatment_scores = _multilabel_scores(self.treatment, monitor_treatment_input, len(self.mappings["treatment_map"]))
                _print_stage_validation_metric(
                    "treatment",
                    treatment_scores,
                    monitor_labels["treatment"],
                    self.mappings,
                    self.args,
                    monitor_split,
                )
                monitor_treatment_features = _scores_to_multilabel_features(treatment_scores, self.args)

            herb_features = _stack_features(base_features, diag_true, syndrome_true, treatment_true)
            _run_with_heartbeat(
                "Sklearn stage 4/4: fitting herb",
                lambda: self.herb.fit(herb_features, labels["herb"]),
                self.args,
            )
            if (
                monitor_features is not None
                and monitor_labels is not None
                and monitor_diag_features is not None
                and monitor_syndrome_features is not None
                and monitor_treatment_features is not None
            ):
                monitor_herb_input = _stack_features(
                    monitor_features,
                    monitor_diag_features,
                    monitor_syndrome_features,
                    monitor_treatment_features,
                )
                herb_scores = _multilabel_scores(self.herb, monitor_herb_input, len(self.mappings["herb_map"]))
                _print_stage_validation_metric("herb", herb_scores, monitor_labels["herb"], self.mappings, self.args, monitor_split)
        return self

    def predict_scores(
        self,
        base_features: sparse.csr_matrix,
        labels: dict[str, np.ndarray],
        mode: str,
    ) -> dict[str, np.ndarray]:
        diag_scores = _multiclass_scores(self.diag, base_features, self.num_diag)
        if mode == "oracle":
            diag_features = _one_hot(labels["diag"], self.num_diag)
        else:
            diag_features = _scores_to_multiclass_features(diag_scores, self.args)

        syndrome_features = _stack_features(base_features, diag_features)
        syndrome_scores = _multiclass_scores(self.syndrome, syndrome_features, self.num_syndrome)
        if mode == "oracle":
            syndrome_chain_features = _one_hot(labels["syndrome"], self.num_syndrome)
        else:
            syndrome_chain_features = _scores_to_multiclass_features(syndrome_scores, self.args)

        treatment_features = _stack_features(base_features, diag_features, syndrome_chain_features)
        treatment_scores = _multilabel_scores(self.treatment, treatment_features, len(self.mappings["treatment_map"]))
        if mode == "oracle":
            treatment_chain_features = _as_sparse(labels["treatment"])
        else:
            treatment_chain_features = _scores_to_multilabel_features(treatment_scores, self.args)

        herb_features = _stack_features(base_features, diag_features, syndrome_chain_features, treatment_chain_features)
        herb_scores = _multilabel_scores(self.herb, herb_features, len(self.mappings["herb_map"]))
        return {
            "diag": diag_scores,
            "syndrome": syndrome_scores,
            "treatment": treatment_scores,
            "herb": herb_scores,
        }


def _result_row(
    model_name: str,
    valid_oracle_metrics: dict[str, dict[str, float]],
    valid_predicted_metrics: dict[str, dict[str, float]],
    test_oracle_metrics: dict[str, dict[str, float]],
    test_predicted_metrics: dict[str, dict[str, float]],
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "status": "ok",
        **flatten_metrics("valid_oracle", valid_oracle_metrics),
        **flatten_metrics("valid_predicted", valid_predicted_metrics),
        **flatten_metrics("test_oracle", test_oracle_metrics),
        **flatten_metrics("test_predicted", test_predicted_metrics),
    }


def _save_metrics_payloads(
    model_dir: Path,
    protocol: dict[str, Any],
    valid_oracle_metrics: dict[str, dict[str, float]],
    valid_predicted_metrics: dict[str, dict[str, float]],
    test_oracle_metrics: dict[str, dict[str, float]],
    test_predicted_metrics: dict[str, dict[str, float]],
    valid_losses: tuple[float | None, float | None] = (None, None),
) -> None:
    valid_payload = {
        "protocol": protocol,
        "summary": {
            "oracle_chain_score": compute_chain_score(valid_oracle_metrics),
            "predicted_chain_score": compute_chain_score(valid_predicted_metrics),
        },
        "oracle": valid_oracle_metrics,
        "predicted": valid_predicted_metrics,
        "losses": {"oracle": _safe_float(valid_losses[0]), "predicted": _safe_float(valid_losses[1])},
    }
    test_payload = {
        "protocol": protocol,
        "summary": {
            "oracle_chain_score": compute_chain_score(test_oracle_metrics),
            "predicted_chain_score": compute_chain_score(test_predicted_metrics),
        },
        "oracle": test_oracle_metrics,
        "predicted": test_predicted_metrics,
    }
    save_json(model_dir / "valid_metrics.json", valid_payload)
    save_json(model_dir / "test_metrics.json", test_payload)


def _print_fit_validation_summary(
    oracle_metrics: dict[str, dict[str, float]],
    predicted_metrics: dict[str, dict[str, float]],
) -> None:
    print("\n--- Validation ---")
    print(f"valid_predicted_chain_score={compute_chain_score(predicted_metrics):.4f}")
    print_score_lines("  Oracle", oracle_metrics)
    print_score_lines("  Predicted", predicted_metrics)


def _print_final_test_tables(
    oracle_metrics: dict[str, dict[str, float]],
    predicted_metrics: dict[str, dict[str, float]],
) -> None:
    print("\n--- Final Testing ---")
    print("\n--- Oracle Chain Evaluation on Test Set ---")
    print_metrics_table("Oracle Chain", oracle_metrics)
    print("\n--- Predicted Chain Evaluation on Test Set ---")
    print_metrics_table("Predicted Chain", predicted_metrics)


def _make_torch_loaders(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: Any | None,
    mappings: dict[str, Any],
    token_map: dict[int, int],
    args: argparse.Namespace,
    include_text: bool,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    pin_memory = bool(args.pin_memory and device.type == "cuda")
    datasets = [
        TCMComparisonDataset(frame, tokenizer, mappings, token_map, args, include_text=include_text)
        for frame in (train_df, valid_df, test_df)
    ]
    return (
        DataLoader(
            datasets[0],
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=int(args.num_workers),
            collate_fn=comparison_collate_fn,
            pin_memory=pin_memory,
        ),
        DataLoader(
            datasets[1],
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            collate_fn=comparison_collate_fn,
            pin_memory=pin_memory,
        ),
        DataLoader(
            datasets[2],
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            collate_fn=comparison_collate_fn,
            pin_memory=pin_memory,
        ),
    )


def _configure_torch_optimizer(model: torch.nn.Module, args: argparse.Namespace, model_name: str):
    transformer_params = []
    other_params = []
    use_pretrained_transformer_lr = model_name in {"MacBERTChain", "MMoEChain", "PLEChain"}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if use_pretrained_transformer_lr and name.startswith("encoder.transformer."):
            transformer_params.append(param)
        else:
            other_params.append(param)
    groups = []
    if transformer_params:
        groups.append({"params": transformer_params, "lr": float(args.learning_rate), "name": "transformer"})
    if other_params:
        groups.append({"params": other_params, "lr": float(args.head_learning_rate), "name": "comparison_head"})
    print("Optimizer parameter groups: " + ", ".join(f"{group['name']}={len(group['params'])}" for group in groups))
    return torch.optim.AdamW(groups, weight_decay=float(args.weight_decay))


def _linear_schedule_with_warmup(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    try:
        from transformers import get_linear_schedule_with_warmup
    except ModuleNotFoundError:
        def schedule(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            remaining = max(0, total_steps - step)
            return float(remaining) / float(max(1, total_steps - warmup_steps))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def run_torch_model(
    model_name: str,
    args: argparse.Namespace,
    mappings: dict[str, Any],
    token_map: dict[int, int],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: torch.device,
    model_dir: Path,
    tokenizer_cache: dict[str, Any],
    graph_data: Any | None,
) -> dict[str, Any]:
    spec = BASELINE_SPECS[model_name]
    include_text = model_name in TEXT_TORCH_MODELS
    tokenizer = None
    tokenizer_vocab_size = None
    if include_text:
        try:
            from transformers import AutoTokenizer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(f"{model_name} requires `transformers` for text tokenization.") from exc
        if "text" not in tokenizer_cache:
            tokenizer_cache["text"] = AutoTokenizer.from_pretrained(
                args.bert_model_name,
                local_files_only=args.local_files_only,
            )
        tokenizer = tokenizer_cache["text"]
        tokenizer_vocab_size = int(getattr(tokenizer, "vocab_size", len(tokenizer)))

    train_loader, valid_loader, test_loader = _make_torch_loaders(
        train_df,
        valid_df,
        test_df,
        tokenizer,
        mappings,
        token_map,
        args,
        include_text,
        device,
    )
    model = build_torch_model(
        model_name,
        args,
        mappings,
        token_map,
        tokenizer_vocab_size=tokenizer_vocab_size,
        graph_data=graph_data,
    ).to(device)
    loss_fns = make_loss_functions(args, device)
    optimizer = _configure_torch_optimizer(model, args, model_name)
    total_steps = max(1, math.ceil(len(train_loader) / int(args.gradient_accumulation_steps)) * int(args.epochs))
    scheduler = _linear_schedule_with_warmup(optimizer, int(total_steps * float(args.warmup_ratio)), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.use_amp and device.type == "cuda"))
    protocol = protocol_dict(args, spec)
    print(f"Evaluation protocol: {protocol}")
    print(f"Training regime: Torch mini-batch on {device}; epochs={args.epochs}, batch_size={args.batch_size}.")
    save_json(
        model_dir / "config.json",
        {
            "args": vars(args),
            "protocol": protocol,
            "graph_stats": graph_data.stats if graph_data is not None else None,
        },
    )

    best_score = -float("inf")
    best_loss = float("inf")
    best_path = model_dir / "best_model.pt"
    training_log = []
    epochs_without_improvement = 0
    for epoch in range(1, int(args.epochs) + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        train_loss = train_torch_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, loss_fns, args)
        valid_oracle_loss, valid_oracle_metrics = evaluate_torch_chain(
            model, valid_loader, device, loss_fns, mappings, args, mode="oracle"
        )
        valid_predicted_loss, valid_predicted_metrics = evaluate_torch_chain(
            model, valid_loader, device, loss_fns, mappings, args, mode="predict"
        )
        print_epoch_summary(
            epoch,
            args,
            train_loss,
            valid_oracle_loss,
            valid_oracle_metrics,
            valid_predicted_loss,
            valid_predicted_metrics,
        )
        training_log.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_oracle_loss": valid_oracle_loss,
                "valid_predicted_loss": valid_predicted_loss,
                **flatten_metrics("valid_oracle", valid_oracle_metrics),
                **flatten_metrics("valid_predicted", valid_predicted_metrics),
            }
        )
        pd.DataFrame(training_log).to_csv(model_dir / "training_log.csv", index=False, encoding="utf-8-sig")
        predicted_score = compute_chain_score(valid_predicted_metrics)
        if predicted_score > best_score + float(args.min_delta):
            best_score = predicted_score
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_path)
            print(f"Best model saved to {best_path} (valid_predicted_chain_score={best_score:.4f})")
        else:
            epochs_without_improvement += 1
        if valid_oracle_loss < best_loss:
            best_loss = valid_oracle_loss
            torch.save(model.state_dict(), model_dir / "best_loss_model.pt")
        if int(args.early_stop_patience) > 0 and epochs_without_improvement >= int(args.early_stop_patience):
            print(f"{model_name} early stopped at epoch {epoch}.")
            break

    print(f"Training log saved to {model_dir / 'training_log.csv'}")
    model.load_state_dict(torch.load(best_path, map_location=device))
    valid_oracle_loss, valid_oracle_metrics = evaluate_torch_chain(model, valid_loader, device, loss_fns, mappings, args, "oracle")
    valid_predicted_loss, valid_predicted_metrics = evaluate_torch_chain(
        model, valid_loader, device, loss_fns, mappings, args, "predict"
    )
    _, test_oracle_metrics = evaluate_torch_chain(model, test_loader, device, loss_fns, mappings, args, "oracle")
    _, test_predicted_metrics = evaluate_torch_chain(model, test_loader, device, loss_fns, mappings, args, "predict")
    _save_metrics_payloads(
        model_dir,
        protocol,
        valid_oracle_metrics,
        valid_predicted_metrics,
        test_oracle_metrics,
        test_predicted_metrics,
        valid_losses=(valid_oracle_loss, valid_predicted_loss),
    )
    _print_final_test_tables(test_oracle_metrics, test_predicted_metrics)
    return _result_row(model_name, valid_oracle_metrics, valid_predicted_metrics, test_oracle_metrics, test_predicted_metrics)


def _load_bge(args: argparse.Namespace, device: torch.device):
    try:
        from transformers import AutoModel, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("BGEM3LRChain requires `transformers`.") from exc
    tokenizer = AutoTokenizer.from_pretrained(args.bge_model_name, local_files_only=args.local_files_only)
    model = AutoModel.from_pretrained(args.bge_model_name, local_files_only=args.local_files_only).to(device)
    model.eval()
    return tokenizer, model


def _bge_embeddings(documents: list[str], args: argparse.Namespace, device: torch.device, tokenizer: Any, model: Any) -> np.ndarray:
    embeddings = []
    for start in tqdm(range(0, len(documents), int(args.bge_batch_size)), desc="Encoding BGE-M3", leave=False):
        encoded = tokenizer(
            documents[start : start + int(args.bge_batch_size)],
            padding=True,
            truncation=True,
            max_length=int(args.bge_max_length),
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            states = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        embeddings.append(pooled.detach().cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def run_bge_model(
    args: argparse.Namespace,
    mappings: dict[str, Any],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: torch.device,
    model_dir: Path,
) -> dict[str, Any]:
    model_name = "BGEM3LRChain"
    protocol = protocol_dict(args, BASELINE_SPECS[model_name])
    print(f"Evaluation protocol: {protocol}")
    print(f"Training regime: BGE batch encoding on {device}, then sklearn CPU stagewise LR fit.")
    tokenizer, encoder = _load_bge(args, device)
    train_features = _as_sparse(
        _bge_embeddings(bge_documents(train_df, args, progress_desc="Preparing BGE train documents"), args, device, tokenizer, encoder)
    )
    valid_features = _as_sparse(
        _bge_embeddings(bge_documents(valid_df, args, progress_desc="Preparing BGE valid documents"), args, device, tokenizer, encoder)
    )
    test_features = _as_sparse(
        _bge_embeddings(bge_documents(test_df, args, progress_desc="Preparing BGE test documents"), args, device, tokenizer, encoder)
    )
    train_labels = dataframe_label_arrays(train_df, mappings, args)
    valid_labels = dataframe_label_arrays(valid_df, mappings, args)
    test_labels = dataframe_label_arrays(test_df, mappings, args)
    print("\n--- Stagewise LR Fit ---")
    model = LogisticRegressionChain(mappings, args).fit(
        train_features,
        train_labels,
        monitor_features=valid_features,
        monitor_labels=valid_labels,
    )
    with open(model_dir / "best_model.pkl", "wb") as handle:
        pickle.dump({"model": model, "bge_model_name": args.bge_model_name}, handle)
    valid_oracle_metrics = compute_metrics_from_arrays(model.predict_scores(valid_features, valid_labels, "oracle"), valid_labels, mappings, args)
    valid_predicted_metrics = compute_metrics_from_arrays(
        model.predict_scores(valid_features, valid_labels, "predict"), valid_labels, mappings, args
    )
    test_oracle_metrics = compute_metrics_from_arrays(model.predict_scores(test_features, test_labels, "oracle"), test_labels, mappings, args)
    test_predicted_metrics = compute_metrics_from_arrays(
        model.predict_scores(test_features, test_labels, "predict"), test_labels, mappings, args
    )
    _print_fit_validation_summary(valid_oracle_metrics, valid_predicted_metrics)
    training_log_path = model_dir / "training_log.csv"
    pd.DataFrame(
        [
            {
                "epoch": 1,
                "train_loss": np.nan,
                "valid_oracle_loss": np.nan,
                "valid_predicted_loss": np.nan,
                **flatten_metrics("valid_oracle", valid_oracle_metrics),
                **flatten_metrics("valid_predicted", valid_predicted_metrics),
            }
        ]
    ).to_csv(training_log_path, index=False, encoding="utf-8-sig")
    print(f"Training log saved to {training_log_path}")
    save_json(model_dir / "config.json", {"args": vars(args), "protocol": protocol})
    _save_metrics_payloads(model_dir, protocol, valid_oracle_metrics, valid_predicted_metrics, test_oracle_metrics, test_predicted_metrics)
    _print_final_test_tables(test_oracle_metrics, test_predicted_metrics)
    return _result_row(model_name, valid_oracle_metrics, valid_predicted_metrics, test_oracle_metrics, test_predicted_metrics)


def build_report(results_df: pd.DataFrame, args: argparse.Namespace) -> str:
    lines = [
        "# AAAI TCM chain baseline comparison",
        "",
        f"Data: `{args.data_path}`",
        "Primary ranking: `test_predicted_chain_score`",
        "Rules used by baselines: false",
        "",
    ]
    ok = results_df[results_df["status"] == "ok"].copy() if not results_df.empty else pd.DataFrame()
    if ok.empty:
        lines.extend(["No trainable baseline finished successfully.", ""])
    else:
        columns = [
            "model_name",
            "test_predicted_chain_score",
            "test_predicted_diag_score",
            "test_predicted_syndrome_score",
            "test_predicted_treatment_score",
            "test_predicted_herb_score",
            "test_oracle_chain_score",
        ]
        lines.extend(["## Finished models", "", "```text", ok.sort_values("test_predicted_chain_score", ascending=False)[columns].to_string(index=False), "```", ""])
    pending = results_df[results_df["status"] != "ok"] if not results_df.empty else pd.DataFrame()
    if not pending.empty:
        lines.extend(["## Non-ok models", "", "```text", pending[["model_name", "status"]].to_string(index=False), "```", ""])
    return "\n".join(lines)


def run_one_model(
    model_name: str,
    args: argparse.Namespace,
    mappings: dict[str, Any],
    token_map: dict[int, int],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: torch.device,
    output_dir: Path,
    tokenizer_cache: dict[str, Any],
    graph_data: Any | None,
) -> dict[str, Any]:
    model_dir = output_dir / "models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    spec = BASELINE_SPECS[model_name]
    if spec.runner in {"torch", "torch_graph"}:
        return run_torch_model(
            model_name,
            args,
            mappings,
            token_map,
            train_df,
            valid_df,
            test_df,
            device,
            model_dir,
            tokenizer_cache,
            graph_data if spec.runner == "torch_graph" else None,
        )
    if spec.runner == "bge":
        return run_bge_model(args, mappings, train_df, valid_df, test_df, device, model_dir)
    raise ValueError(f"Unsupported runner {spec.runner} for {model_name}")


def main(args: argparse.Namespace) -> None:
    set_seed(int(args.random_seed))
    print_gpu_info()
    device = get_device(args.gpu_id)
    output_dir = Path(args.output_dir)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.data_path}...")
    dataframe = read_csv_with_encoding_fallback(args.data_path)
    if int(args.max_rows) > 0:
        dataframe = dataframe.head(int(args.max_rows)).copy()
        print(f"Debug max_rows enabled: {len(dataframe)}")
    mappings = build_label_mappings(
        dataframe,
        args.tcm_diag_col,
        args.syndrome_col,
        args.treatment_col,
        args.herb_col,
        args.western_diag_col,
        mapping_mode=args.label_mapping_mode,
    )
    token_map = build_token_map(dataframe, args.token_col)
    if "split" in dataframe.columns and not args.max_rows:
        train_df = dataframe[dataframe["split"] == "train"].copy()
        valid_df = dataframe[dataframe["split"] == "valid"].copy()
        test_df = dataframe[dataframe["split"] == "test"].copy()
        if min(len(train_df), len(valid_df), len(test_df)) == 0:
            raise ValueError("The fixed split column must contain train, valid, and test rows.")
    else:
        train_df, test_df = train_test_split(dataframe, test_size=0.25, random_state=int(args.random_seed))
        valid_df, test_df = train_test_split(test_df, test_size=0.5, random_state=int(args.random_seed))
    model_names = select_model_names(args)
    print(f"Dataset split: Train={len(train_df)}, Valid={len(valid_df)}, Test={len(test_df)}")
    if any(BASELINE_SPECS[name].runner in {"torch", "torch_graph"} for name in model_names):
        attach_training_statistics(args, train_df, mappings)
        print(f"Age normalization: mean={args.age_mean:.4f}, std={args.age_std:.4f}")
    print_execution_plan(model_names, args, device)
    save_json(output_dir / "label_mappings.json", mappings)
    save_json(output_dir / "token_map.json", token_map)
    save_json(
        output_dir / "experiment_config.json",
        {
            "args": vars(args),
            "model_names": model_names,
            "baseline_specs": {name: vars(spec) for name, spec in BASELINE_SPECS.items()},
        },
    )

    graph_data = None
    if any(BASELINE_SPECS[name].runner == "torch_graph" for name in model_names):
        graph_data = build_train_cooccurrence_graph(train_df, mappings, token_map, args)
        save_json(output_dir / "train_cooccurrence_graph_stats.json", graph_data.stats)
    tokenizer_cache: dict[str, Any] = {}
    results = []
    for model_name in model_names:
        print(f"\n========== Running {model_name} ==========")
        try:
            result = run_one_model(
                model_name,
                args,
                mappings,
                token_map,
                train_df,
                valid_df,
                test_df,
                device,
                output_dir,
                tokenizer_cache,
                graph_data,
            )
        except Exception as exc:
            model_dir = output_dir / "models" / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"[{model_name}] failed: {exc}")
            if len(model_names) == 1:
                raise
            result = {"model_name": model_name, "status": "failed", "error": str(exc)}
        spec = BASELINE_SPECS[model_name]
        result = {
            "model_name": model_name,
            "setting": spec.setting,
            "runner": spec.runner,
            "training_regime": _training_regime(spec),
            **result,
        }
        results.append(result)
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_dir / "all_model_results.csv", index=False, encoding="utf-8-sig")
        save_json(output_dir / "all_model_results.json", results)

    results_df = pd.DataFrame(results)
    report = build_report(results_df, args)
    (output_dir / "final_report.md").write_text(report, encoding="utf-8")
    print(f"\nFinished. Outputs saved to {output_dir}")
    print(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AAAI TCM chain baseline comparisons without author symbolic rules.")
    parser.add_argument("--data_path", type=str, default=str(PROJECT_ROOT / "data" / "tcm_benchmark.csv"))
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "baselines"))
    parser.add_argument("--model_list", type=str, default="")
    parser.add_argument(
        "--model_set",
        choices=MODEL_SET_CHOICES,
        default="neural",
        help="Default to GPU/batched Torch baselines. Use all to include BGE-M3-LR.",
    )
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

    parser.add_argument("--bert_model_name", type=str, default="hfl/chinese-macbert-base")
    parser.add_argument("--bge_model_name", type=str, default="BAAI/bge-m3")
    parser.add_argument("--allow_model_download", dest="local_files_only", action="store_false")
    parser.set_defaults(local_files_only=True)
    parser.add_argument("--freeze_bert", action="store_true")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--text_encoder_dim", type=int, default=256)
    parser.add_argument("--textcnn_kernel_sizes", type=str, default="3,4,5")
    parser.add_argument("--lexicon_encoder_dim", type=int, default=128)
    parser.add_argument("--lexicon_num_heads", type=int, default=4)
    parser.add_argument("--lexicon_num_layers", type=int, default=2)
    parser.add_argument("--mmoe_num_experts", type=int, default=4)
    parser.add_argument("--ple_shared_experts", type=int, default=2)
    parser.add_argument("--ple_task_experts", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--head_learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument(
        "--progress_heartbeat_seconds",
        type=float,
        default=30.0,
        help="Print a still-running heartbeat for long non-batched preprocessing and sklearn fits. Use 0 to disable.",
    )
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--early_stop_patience", type=int, default=0)
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

    parser.add_argument("--sklearn_max_iter", type=int, default=300)
    parser.add_argument("--sklearn_class_weight_balanced", action="store_true")
    parser.add_argument("--logistic_c", type=float, default=1.0)
    parser.add_argument("--sklearn_n_jobs", type=int, default=1)
    parser.add_argument("--bge_batch_size", type=int, default=16)
    parser.add_argument("--bge_max_length", type=int, default=128)

    parser.add_argument("--graph_dim", type=int, default=128)
    parser.add_argument("--graph_top_k_per_source", type=int, default=32)
    parser.add_argument("--graph_min_edge_count", type=int, default=2)
    parser.add_argument("--graph_max_tokens_per_case", type=int, default=64)

    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
