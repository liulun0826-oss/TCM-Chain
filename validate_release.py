"""Run non-training integrity checks for the benchmark release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
REQUIRED_FILES = (
    "README.md",
    "BENCHMARK_PROTOCOL.txt",
    "DATA_CARD.txt",
    "config/benchmark.json",
    "train.py",
    "requirements.txt",
    "prepare_release_data.py",
    "validate_release.py",
    "tests/test_release.py",
    "tests/test_ablation_variants.py",
    "ablation_variants/__init__.py",
    "ablation_variants/README.md",
    "ablation_variants/train_ablation.py",
    "ablation_variants/variants.py",
    "ablation_variants/configs/OnlyN.json",
    "ablation_variants/configs/OnlyS.json",
    "ablation_variants/configs/wo_DiscrepancyGate.json",
    "ablation_variants/configs/wo_FRules.json",
    "ablation_variants/configs/wo_HRules.json",
    "ablation_variants/configs/wo_CRules.json",
    "ablation_variants/configs/wo_RandomR.json",
    "ablation_variants/configs/wo_NegRules.json",
    "src/__init__.py",
    "baseline/__init__.py",
    "baseline/README.md",
    "baseline/launcher.py",
    "baseline/train.py",
    "baseline/models.py",
    "baseline/utils.py",
    "baseline/MacBERT.py",
    "baseline/PLE.py",
    "baseline/MMoE.py",
    "baseline/TextCNN.py",
    "baseline/BiLSTM-Attn.py",
    "baseline/GCN.py",
    "baseline/LightXML.py",
    "baseline/GAT.py",
    "baseline/R-GCN.py",
    "baseline/HAN.py",
    "baseline/HGT.py",
    "baseline/Lexicon-Transformer.py",
    "baseline/BGE-M3-LR.py",
    "data/clinical_data_access_notes.txt",
    "data/index_data/initial_diagnosis_index.csv",
    "data/index_data/tcm_diagnosis_index.csv",
    "data/index_data/syndrome_index.csv",
    "data/index_data/treatment_principle_index.csv",
    "data/index_data/herb_index.csv",
    "data/index_data/medical_text_lexicon_index.csv",
    "rules/llm_audited_rules.csv",
    "rules/high_order_rules.csv",
    "rules/path_rules.csv",
    "rules/treatment_pair_rules.csv",
    "rules/herb_pair_rules.csv",
    "src/backbone.py",
    "src/dataset.py",
    "src/metrics.py",
    "src/dinsr.py",
    "src/neural_symbolic_dataset.py",
    "src/symbolic_dataset.py",
    "src/utils.py",
)
INDEX_FILES = tuple(relative for relative in REQUIRED_FILES if relative.startswith("data/index_data/"))
RESTRICTED_DATA_FILES = (
    "data/tcm_benchmark.csv",
    "data/deidentification_report.json",
)
BENCHMARK_ROWS = 75315
BENCHMARK_SPLIT_COUNTS = {"train": 56486, "valid": 9414, "test": 9415}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(update_manifest: bool = False) -> None:
    missing = [relative for relative in REQUIRED_FILES if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing release files: {missing}")
    leaked_restricted = [relative for relative in RESTRICTED_DATA_FILES if (ROOT / relative).exists()]
    if leaked_restricted:
        raise ValueError(f"Restricted clinical data files must not be in the public release: {leaked_restricted}")
    forbidden_extensions = {".pdf", ".pt", ".pth", ".ckpt"}
    forbidden_files = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_extensions
    ]
    if forbidden_files:
        raise ValueError(f"Forbidden release artifacts found: {forbidden_files}")

    for source in [
        ROOT / "train.py",
        *(ROOT / "src").glob("*.py"),
        *(ROOT / "baseline").glob("*.py"),
        *(ROOT / "ablation_variants").glob("*.py"),
    ]:
        compile(source.read_text(encoding="utf-8"), str(source), "exec")

    for relative in INDEX_FILES:
        index_frame = pd.read_csv(ROOT / relative, encoding="utf-8-sig", nrows=5)
        expected_index_columns = ["id", "name", "count"]
        if list(index_frame.columns) != expected_index_columns:
            raise ValueError(f"Unexpected index columns in {relative}: {list(index_frame.columns)}")

    for relative in REQUIRED_FILES:
        if relative.startswith("rules/"):
            with (ROOT / relative).open("r", encoding="utf-8-sig", newline="") as handle:
                header = next(csv.reader(handle), [])
            if not header:
                raise ValueError(f"Rule file has no CSV header: {relative}")

    release_files = sorted(
        path for path in ROOT.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.name != "MANIFEST.json"
    )
    manifest = {
        "release": "DiNSR_Benchmark",
        "public_release": True,
        "restricted_clinical_data_included": False,
        "benchmark_rows_when_authorized_data_is_generated": BENCHMARK_ROWS,
        "split_counts_when_authorized_data_is_generated": BENCHMARK_SPLIT_COUNTS,
        "files": {
            path.relative_to(ROOT).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in release_files
        },
    }
    manifest_path = ROOT / "MANIFEST.json"
    if update_manifest:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
        print("Manifest updated; release checks passed.")
        return
    if not manifest_path.is_file():
        raise FileNotFoundError("MANIFEST.json is missing; run validate_release.py --update-manifest.")
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if recorded != manifest:
        raise ValueError("MANIFEST.json is out of date; review changes, then run validate_release.py --update-manifest.")
    print("Release checks and manifest verification passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-manifest", action="store_true")
    main(parser.parse_args().update_manifest)
