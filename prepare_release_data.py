"""Create the fixed-split, explicitly de-identified benchmark table."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from unidecode import unidecode


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "data" / "tcm_benchmark.csv"

RAW_PRESCRIPTION_ID_COL = "\u5904\u65b9\u53f7"
RAW_PATIENT_NAME_COL = "\u59d3\u540d"
RAW_MEDICAL_RECORD_SUMMARY_COL = "\u75c5\u5386\u6c47\u603b"
RAW_SEX_COL = "\u6027\u522b"
RAW_AGE_COL = "\u5e74\u9f84"
RAW_INITIAL_DIAGNOSIS_COL = "\u521d\u6b65\u8bca\u65ad"
RAW_TCM_DIAGNOSIS_COL = "\u4e2d\u533b\u8bca\u65ad"
RAW_SYNDROME_COL = "\u8bc1\u578b"
RAW_TREATMENT_COL = "\u6cbb\u5219\u6cbb\u6cd5"
RAW_HERB_COL = "\u836f\u540d\u4e0e\u5355\u5e16\u91cd\u91cf"
RAW_TEXT_LEXICON_COL = "\u75c5\u5386\u6587\u672c\u8bcd\u6e90"
RAW_ID_COL = "\u7f16\u53f7"
RAW_NAME_COL = "\u540d\u79f0"
RAW_COUNT_COL = "\u51fa\u73b0\u6b21\u6570"
RAW_HERB_NAME_COL = "\u4e2d\u836f\u540d\u79f0"

IDENTIFIER_COLUMNS = (RAW_PRESCRIPTION_ID_COL, RAW_PATIENT_NAME_COL)
IDENTIFIER_REPORT_COLUMNS = ("prescription_id", "patient_name")
FREE_TEXT_COLUMNS = (RAW_MEDICAL_RECORD_SUMMARY_COL,)
CORE_RELEASE_COLUMNS = (
    RAW_SEX_COL,
    RAW_AGE_COL,
    RAW_INITIAL_DIAGNOSIS_COL,
    RAW_TCM_DIAGNOSIS_COL,
    RAW_SYNDROME_COL,
    RAW_TREATMENT_COL,
    RAW_HERB_COL,
    RAW_MEDICAL_RECORD_SUMMARY_COL,
    RAW_TEXT_LEXICON_COL,
)
PUBLIC_COLUMN_MAP = {
    RAW_SEX_COL: "sex",
    RAW_AGE_COL: "age",
    RAW_INITIAL_DIAGNOSIS_COL: "initial_diagnosis",
    RAW_TCM_DIAGNOSIS_COL: "tcm_diagnosis",
    RAW_SYNDROME_COL: "syndrome",
    RAW_TREATMENT_COL: "treatment_principles",
    RAW_HERB_COL: "herbs",
    RAW_MEDICAL_RECORD_SUMMARY_COL: "medical_record_summary",
    RAW_TEXT_LEXICON_COL: "medical_text_lexicon",
}
PUBLIC_RELEASE_COLUMNS = tuple(PUBLIC_COLUMN_MAP[column] for column in CORE_RELEASE_COLUMNS)
INDEX_COLUMNS = ("id", "name", "count")
INDEX_EXPORTS = (
    (
        "\u8bca\u65ad.csv",
        "diagnosis.csv",
        "initial_diagnosis_index.csv",
        {RAW_ID_COL: "id", RAW_NAME_COL: "name", RAW_COUNT_COL: "count"},
    ),
    (
        "\u4e2d\u533b\u8bca\u65ad.csv",
        "tcm_diagnosis.csv",
        "tcm_diagnosis_index.csv",
        {RAW_ID_COL: "id", RAW_NAME_COL: "name", RAW_COUNT_COL: "count"},
    ),
    (
        "\u4e2d\u533b\u8bc1\u578b.csv",
        "tcm_syndrome.csv",
        "syndrome_index.csv",
        {RAW_ID_COL: "id", RAW_NAME_COL: "name", RAW_COUNT_COL: "count"},
    ),
    (
        "\u4e2d\u533b\u6cbb\u5219_frequency_rank.csv",
        "tcm_treatment_frequency_rank.csv",
        "treatment_principle_index.csv",
        {"ID": "id", "Word": "name", "Count": "count"},
    ),
    (
        "\u5904\u65b9\u5185\u5bb9.csv",
        "prescription_content.csv",
        "herb_index.csv",
        {RAW_ID_COL: "id", RAW_HERB_NAME_COL: "name", RAW_COUNT_COL: "count"},
    ),
    (
        "\u75c5\u5386\u6587\u672c\u8bcd\u6e90.csv",
        "medical_text_lexicon.csv",
        "medical_text_lexicon_index.csv",
        {RAW_ID_COL: "id", RAW_NAME_COL: "name", RAW_COUNT_COL: "count"},
    ),
)
SENSITIVE_PATTERNS = {
    "phone_numbers": (r"(?<!\d)1[3-9]\d{9}(?!\d)", "[PHONE_NUMBER]"),
    "cn_identity_numbers": (r"(?<!\d)\d{17}[0-9Xx](?!\d)", "[ID_NUMBER]"),
    "email_addresses": (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]"),
}


def read_csv(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Unable to read {path}: {last_error}")


def public_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unidecode(str(value)).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,;:)\]\}])", r"\1", text)
    text = re.sub(r"([(\[\{])\s+", r"\1", text)
    text = text.replace(" | ", "|").replace("| ", "|").replace(" |", "|")
    return text


def replace_row_identifier(text: object, identifier: object, replacement: str) -> tuple[object, int]:
    if pd.isna(text) or pd.isna(identifier):
        return text, 0
    value = str(identifier).strip()
    if not value or value.lower() == "nan":
        return text, 0
    original = str(text)
    count = original.count(value)
    return original.replace(value, replacement), count


def fixed_splits(size: int, seed: int) -> np.ndarray:
    indices = np.arange(size)
    train_indices, held_out = train_test_split(indices, test_size=0.25, random_state=seed)
    valid_indices, test_indices = train_test_split(held_out, test_size=0.5, random_state=seed)
    split = np.full(size, "", dtype=object)
    split[train_indices] = "train"
    split[valid_indices] = "valid"
    split[test_indices] = "test"
    return split


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_index_tables(source_dir: Path, output_dir: Path) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    for source_name, source_label, target_name, column_map in INDEX_EXPORTS:
        source_path = source_dir / source_name
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing index source file: {source_path}")
        frame = read_csv(source_path)
        missing_columns = [column for column in column_map if column not in frame.columns]
        if missing_columns:
            missing_labels = [public_name(column) for column in missing_columns]
            raise ValueError(f"Missing index columns in {source_path}: {missing_labels}")
        normalized = frame.loc[:, list(column_map.keys())].rename(columns=column_map)
        normalized = normalized.loc[:, list(INDEX_COLUMNS)]
        normalized["name"] = normalized["name"].map(public_name)
        target_path = output_dir / target_name
        normalized.to_csv(target_path, index=False, encoding="utf-8-sig")
        exported.append(
            {
                "source_file": source_label,
                "release_file": f"index_data/{target_name}",
                "rows": int(len(normalized)),
                "sha256": sha256(target_path),
            }
        )
    return exported


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--index_source_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = read_csv(args.source)
    missing = [column for column in IDENTIFIER_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing identifier columns: {[public_name(column) for column in missing]}")
    missing_core = [column for column in CORE_RELEASE_COLUMNS if column not in frame.columns]
    if missing_core:
        raise ValueError(f"Missing core release columns: {[public_name(column) for column in missing_core]}")

    replacement_counts = {"name": 0, "prescription_id": 0}
    for column in FREE_TEXT_COLUMNS:
        if column not in frame.columns:
            continue
        replaced_values = []
        for text, name, prescription_id in zip(
            frame[column],
            frame[RAW_PATIENT_NAME_COL],
            frame[RAW_PRESCRIPTION_ID_COL],
        ):
            value, name_count = replace_row_identifier(text, name, "[NAME]")
            value, prescription_count = replace_row_identifier(value, prescription_id, "[CASE_ID]")
            replacement_counts["name"] += name_count
            replacement_counts["prescription_id"] += prescription_count
            replaced_values.append(value)
        frame[column] = replaced_values

    pattern_replacement_counts = {name: 0 for name in SENSITIVE_PATTERNS}
    for column in FREE_TEXT_COLUMNS:
        if column not in frame.columns:
            continue
        values = frame[column].fillna("").astype(str)
        for name, (pattern, replacement) in SENSITIVE_PATTERNS.items():
            pattern_replacement_counts[name] += int(values.str.count(pattern).sum())
            values = values.str.replace(pattern, replacement, regex=True)
        frame[column] = values

    source_columns = set(frame.columns)
    dropped_non_core_columns = sorted(
        column
        for column in source_columns.difference(CORE_RELEASE_COLUMNS).difference(IDENTIFIER_COLUMNS)
    )
    dropped_non_core_columns = [public_name(column) for column in dropped_non_core_columns]
    frame = frame.loc[:, list(CORE_RELEASE_COLUMNS)].rename(columns=PUBLIC_COLUMN_MAP).copy()
    frame.insert(0, "split", fixed_splits(len(frame), args.seed))
    frame.insert(0, "case_id", [f"TCM-{index:06d}" for index in range(1, len(frame) + 1)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8-sig")
    index_files = export_index_tables(args.index_source_dir, args.output.parent / "index_data")

    text_columns = [column for column in ("medical_record_summary",) if column in frame.columns]
    combined_text = frame[text_columns].fillna("").astype(str).agg(" ".join, axis=1)
    pattern_counts = {
        name: int(combined_text.str.contains(re.compile(pattern), regex=True).sum())
        for name, (pattern, _) in SENSITIVE_PATTERNS.items()
    }
    digest = sha256(args.output)
    report = {
        "source_rows": len(frame),
        "removed_columns": list(IDENTIFIER_REPORT_COLUMNS),
        "dropped_non_core_columns": dropped_non_core_columns,
        "core_release_columns": ["case_id", "split", *PUBLIC_RELEASE_COLUMNS],
        "index_files": index_files,
        "split_ratio": {"train": 6, "valid": 1, "test": 1},
        "replacement_counts": replacement_counts,
        "pattern_replacement_counts": pattern_replacement_counts,
        "fixed_split_counts": frame["split"].value_counts().sort_index().to_dict(),
        "automated_pattern_audit": pattern_counts,
        "output_sha256": digest,
        "privacy_review_required": True,
    }
    report_path = args.output.with_name("deidentification_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
