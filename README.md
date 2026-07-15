# DiNSR: Four-Stage Neural-Symbolic TCM Recommendation Benchmark

DiNSR is the main model in this release. It uses Chinese MacBERT as the
neural backbone and injects audited first-order rules, high-order combination
rules, path rules, and treatment/herb co-occurrence rules into a four-stage
prediction chain:

```text
medical record -> TCM diagnosis -> syndrome -> treatment principles -> herbs
```

This directory is a minimal public benchmark release. It keeps the DiNSR code,
rule files, index tables, data preparation workflow, configuration, verified
comparison baselines, and release checks. It does not include patient-level clinical data, old
experiment reports, training logs, paper PDFs, ethics scans, or pretrained
checkpoints.

## File Map

| File | Purpose |
|---|---|
| `README.md` | Main project guide. |
| `requirements.txt` | Python dependencies for data preparation and training. |
| `train.py` | Training entry point for DiNSR. |
| `ablation_variants/` | Paper-aligned implementations and configurations for all eight ablation variants. |
| `baseline/` | Runnable comparison models and their usage guide. |
| `validate_release.py` | Release integrity checker for required files, hashes, and public data boundaries. |
| `prepare_release_data.py` | Authorized local preprocessing script for de-identification, fixed splitting, and index export. |
| `BENCHMARK_PROTOCOL.txt` | Unified evaluation protocol. |
| `DATA_CARD.txt` | Dataset card covering tasks, fields, governance, and limits. |
| `MANIFEST.json` | File-size and SHA-256 manifest. |
| `.gitignore` | Ignore rules for model weights, caches, generated data, and outputs. |
| `.gitattributes` | Text normalization rules. |
| `config/benchmark.json` | Benchmark configuration summary. |
| `data/clinical_data_access_notes.txt` | Notes for authorized clinical data generation. |
| `data/index_data/initial_diagnosis_index.csv` | ID-to-name index for initial diagnoses. |
| `data/index_data/tcm_diagnosis_index.csv` | ID-to-name index for TCM diagnoses. |
| `data/index_data/syndrome_index.csv` | ID-to-name index for syndromes. |
| `data/index_data/treatment_principle_index.csv` | ID-to-name index for treatment principles. |
| `data/index_data/herb_index.csv` | ID-to-name index for herbs. |
| `data/index_data/medical_text_lexicon_index.csv` | ID-to-name index for medical-text lexicon tokens. |
| `rules/llm_audited_rules.csv` | Audited first-order positive and negative rule set. |
| `rules/high_order_rules.csv` | Multi-antecedent high-order rule set. |
| `rules/path_rules.csv` | Path rule set connecting diagnoses, syndromes, treatment principles, and herbs. |
| `rules/treatment_pair_rules.csv` | Directed co-occurrence rules between treatment principles. |
| `rules/herb_pair_rules.csv` | Directed co-occurrence rules between herbs. |
| `src/dinsr.py` | Core DiNSR implementation, including rule indexing, symbolic reasoning, and gated fusion. |
| `src/backbone.py` | Four-stage MacBERT neural chain backbone. |
| `src/dataset.py` | Label parsing, label mapping, and base dataset handling. |
| `src/neural_symbolic_dataset.py` | Joint wrapper for neural inputs and symbolic token inputs. |
| `src/symbolic_dataset.py` | Symbolic token parsing, mapping, and collation helpers. |
| `src/metrics.py` | Macro-F1, micro-F1, top-k, and related metrics. |
| `src/utils.py` | Random seed, device, and CSV utility helpers. |
| `src/__init__.py` | Python package marker for the source directory. |
| `tests/test_release.py` | Automated release tests. |

## Model

DiNSR models a four-task TCM recommendation chain:

1. Predict the TCM diagnosis from the medical record.
2. Predict the syndrome from the medical record and diagnosis.
3. Predict one or more treatment principles from upstream information.
4. Predict one or more herbs from the full clinical chain.

The neural branch encodes the medical-record summary with Chinese MacBERT and
combines it with age, sex, and the initial diagnosis. The symbolic branch uses
four knowledge sources: audited first-order rules, multi-antecedent high-order
rules, cross-stage path rules, and directed co-occurrence rules for treatment
principles and herbs.

Rule sources are aggregated with task-specific source attention. A sample-level
gate then controls how strongly symbolic evidence is injected into neural
logits. Training uses teacher forcing with true upstream labels. The primary
test setting is predicted-chain inference, where downstream stages receive only
model-predicted upstream outputs.

## Dataset

This public repository does not include patient-level clinical records at
`data/tcm_benchmark.csv`. The retained ethics material is a
project-application-stage ethics review opinion. To avoid over-interpreting it
as complete public redistribution authorization, the repository publishes code,
rules, index tables, and preparation logic only. Complete patient-level data
may be generated or used locally only after the responsible institution has
confirmed that ethics approval and data-sharing terms cover the use case.

Authorized local generation produces 75,315 rows with a fixed split:

| Split | Rows | Ratio |
|---|---:|---:|
| train | 56,486 | 75% |
| valid | 9,414 | 12.5% |
| test | 9,415 | 12.5% |

The model-facing columns are `sex`, `age`, `initial_diagnosis`,
`medical_record_summary`, and `medical_text_lexicon`. The prediction targets
are `tcm_diagnosis`, `syndrome`, `treatment_principles`, and `herbs`.

The main generated data uses integer IDs for diagnoses, syndromes, treatment
principles, herbs, and lexicon tokens. Public index tables live under
`data/index_data/` and use the shared columns `id`, `name`, and `count`. The
public names are romanized so the release contains no Chinese characters while
preserving stable IDs and frequencies.

`prepare_release_data.py` removes the original patient name and prescription
number, drops redundant raw free-text columns, keeps the model-facing summary,
adds a non-identifying `case_id`, and scans for phone numbers, identity
numbers, and email addresses. Automated processing cannot prove that free
clinical text is fully anonymized. Any public release of complete patient-level
data still requires human privacy review, ethics review, and institutional
authorization.

## Environment

Use Python 3.10 or newer. Install a PyTorch build matching your CUDA
environment when GPU training is desired.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Pre-Training Checks

Run the release checker:

```bash
python validate_release.py
```

Run the unit tests:

```bash
python -m unittest discover -s tests -v
```

A passing check means required public files are present, source files compile,
index tables have the expected English schema, restricted clinical data has not
been mixed into the public directory, and no checkpoint, PDF, or stale
experiment Markdown artifact is present.

## Training

The public repository does not include `data/tcm_benchmark.csv`. Authorized
users should generate it in a private local environment:

```bash
python prepare_release_data.py \
  --source /approved/path/clinical_source.csv \
  --index_source_dir /approved/path/index_tables
```

Then train (model download is enabled by default):

```bash
python train.py --use_amp --pin_memory
```

If Chinese MacBERT is already available locally:

```bash
python train.py \
  --bert_model_name /path/to/chinese-macbert-base \
  --local_files_only \
  --use_amp \
  --pin_memory
```

Remove `--use_amp` and `--pin_memory` for CPU-only execution. Full training on
CPU will be much slower.

Default training settings:

| Parameter | Default |
|---|---:|
| epochs | 50 |
| batch size | 16 |
| random seed | 42 |
| max length | 256 |
| MacBERT learning rate | 2e-5 |
| neural head learning rate | 1e-3 |
| rule learning rate | 5e-4 |
| fusion learning rate | 1e-3 |
| weight decay | 0.01 |
| warmup ratio | 0.1 |

Training output is written to `outputs/dinsr/`. Main artifacts include
`best_model.pt`, `training_log.csv`, `valid_metrics.json`, `test_metrics.json`,
and `gate_statistics.json`. The `outputs/` directory is ignored by Git.

## Ablation Study

The eight variants reported in the paper are implemented under
`ablation_variants/`: `OnlyN`, `OnlyS`, `w/o DiscrepancyGate`, `w/o FRules`,
`w/o HRules`, `w/o CRules`, `w/o RandomR`, and `w/o NegRules`. Each variant
changes only its named mechanism and reuses the full model's split, training
schedule, checkpoint selection rule, and PC/OC evaluation procedures.

Example:

```bash
python ablation_variants/train_ablation.py \
  --variant "w/o DiscrepancyGate" \
  --use_amp \
  --pin_memory
```

See `ablation_variants/README.md` for the controlled intervention associated
with each paper name and the safeguards used for `OnlyS` and randomized-rule
evaluation.

## Benchmark Protocol

Primary evaluation uses predicted-chain results. TCM diagnosis and syndrome use
macro-F1; treatment principles and herbs use micro-F1 at threshold 0.5. The
final `chain_score` is the arithmetic mean of the four primary task metrics.

The best checkpoint is selected only by `valid_predicted_chain_score`. The test
set must not be used for tuning or checkpoint selection. Oracle-chain metrics
are diagnostic only and should not be reported as the main result.

Default ranking diagnostics use diagnosis Top-3, syndrome Top-3, treatment
principle Top-5, and herb Top-13. See `BENCHMARK_PROTOCOL.txt` for the full
protocol.

## Release Notes

- Do not modify the `split` column when reporting comparable benchmark results.
- Do not regenerate rules, vocabularies, or co-occurrence statistics from the
  validation or test split.
- Do not commit `data/tcm_benchmark.csv`, `data/deidentification_report.json`,
  or ethics PDF scans to the public repository.
- Report all four task metrics and predicted-chain `chain_score`.
- This repository does not include a trained checkpoint; training starts from
  pretrained MacBERT.
- Use `MANIFEST.json` to check whether files were corrupted during transfer.
- Add a formal license before public distribution. Without a LICENSE file, the
  code is not automatically open source.
