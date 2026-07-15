from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    jaccard_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
import numpy as np
import torch


def _as_float(value):
    return float(value)


def _sigmoid_numpy(logits):
    return torch.sigmoid(torch.from_numpy(logits)).numpy()


def compute_multiclass_metrics(preds, labels, num_classes, top_k=3):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    acc = accuracy_score(labels_flat, preds_flat)
    macro_precision = precision_score(labels_flat, preds_flat, average='macro', zero_division=0)
    macro_recall = recall_score(labels_flat, preds_flat, average='macro', zero_division=0)
    macro_f1 = f1_score(labels_flat, preds_flat, average='macro', zero_division=0)
    micro_precision = precision_score(labels_flat, preds_flat, average='micro', zero_division=0)
    micro_recall = recall_score(labels_flat, preds_flat, average='micro', zero_division=0)
    micro_f1 = f1_score(labels_flat, preds_flat, average='micro', zero_division=0)

    top_k = min(top_k, num_classes)
    topk_preds = torch.topk(torch.from_numpy(preds), top_k, dim=1).indices.numpy()
    topk_acc = np.mean([1 if labels_flat[i] in topk_preds[i] else 0 for i in range(len(labels_flat))])

    return {
        "accuracy": _as_float(acc),
        "precision": _as_float(micro_precision),
        "recall": _as_float(micro_recall),
        "micro_f1": _as_float(micro_f1),
        "macro_precision": _as_float(macro_precision),
        "macro_recall": _as_float(macro_recall),
        "macro_f1": _as_float(macro_f1),
        "top_k": int(top_k),
        "top_k_accuracy": _as_float(topk_acc),
        "score": _as_float(macro_f1),
    }


def _make_topk_predictions(scores, k):
    k = min(k, scores.shape[1])
    topk_indices = np.argsort(scores, axis=1)[:, -k:]
    preds_topk = np.zeros_like(scores, dtype=int)
    for i in range(len(preds_topk)):
        preds_topk[i, topk_indices[i]] = 1
    return preds_topk, topk_indices


def _ranking_metrics(scores, labels, k):
    k = min(k, scores.shape[1])
    _, topk_indices = _make_topk_predictions(scores, k)
    precision_at_k = []
    recall_at_k = []
    f1_at_k = []
    ndcg_at_k = []

    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    for i in range(len(labels)):
        sample_labels = labels[i]
        hits = sample_labels[topk_indices[i]].astype(float)
        true_count = int(sample_labels.sum())

        p = float(hits.sum() / k)
        r = float(hits.sum() / true_count) if true_count > 0 else 0.0
        f1 = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        dcg = float((hits * discounts).sum())
        ideal_hits = np.sort(sample_labels.astype(float))[::-1][:k]
        idcg = float((ideal_hits * discounts[: len(ideal_hits)]).sum())
        ndcg = dcg / idcg if idcg > 0 else 0.0

        precision_at_k.append(p)
        recall_at_k.append(r)
        f1_at_k.append(f1)
        ndcg_at_k.append(ndcg)

    return {
        "precision_at_k": _as_float(np.mean(precision_at_k)),
        "recall_at_k": _as_float(np.mean(recall_at_k)),
        "f1_at_k": _as_float(np.mean(f1_at_k)),
        "ndcg_at_k": _as_float(np.mean(ndcg_at_k)),
    }


def _prauc_macro(probs, labels):
    scores = []
    for i in range(labels.shape[1]):
        positives = np.unique(labels[:, i])
        if len(positives) < 2:
            continue
        precision, recall, _ = precision_recall_curve(labels[:, i], probs[:, i])
        scores.append(auc(recall, precision))
    return float(np.mean(scores)) if scores else 0.0


def compute_multilabel_metrics(preds, labels, threshold=0.5, rank_k=None):
    # preds: (N, C) logits
    # labels: (N, C) multi-hot

    probs = _sigmoid_numpy(preds)
    preds_bin = (probs >= threshold).astype(int)

    precision = precision_score(labels, preds_bin, average='micro', zero_division=0)
    recall = recall_score(labels, preds_bin, average='micro', zero_division=0)
    micro_f1 = f1_score(labels, preds_bin, average='micro', zero_division=0)
    macro_f1 = f1_score(labels, preds_bin, average='macro', zero_division=0)
    jaccard_micro = jaccard_score(labels, preds_bin, average='micro', zero_division=0)
    jaccard_samples = jaccard_score(labels, preds_bin, average='samples', zero_division=0)

    metrics = {
        "threshold": _as_float(threshold),
        "precision": _as_float(precision),
        "recall": _as_float(recall),
        "micro_f1": _as_float(micro_f1),
        "macro_f1": _as_float(macro_f1),
        "jaccard": _as_float(jaccard_micro),
        "jaccard_samples": _as_float(jaccard_samples),
        "prauc": _as_float(_prauc_macro(probs, labels)),
        "score": _as_float(micro_f1),
    }

    if rank_k is not None and rank_k > 0:
        topk_bin, _ = _make_topk_predictions(preds, rank_k)
        metrics.update({
            "rank_k": int(min(rank_k, preds.shape[1])),
            "topk_micro_f1": _as_float(f1_score(labels, topk_bin, average='micro', zero_division=0)),
            "topk_macro_f1": _as_float(f1_score(labels, topk_bin, average='macro', zero_division=0)),
            "topk_jaccard": _as_float(jaccard_score(labels, topk_bin, average='micro', zero_division=0)),
            **_ranking_metrics(preds, labels, rank_k),
        })

    return metrics
