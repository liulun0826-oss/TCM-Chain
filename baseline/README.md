# Comparison baselines

This directory contains runnable comparison models using the same four-stage
task definition and evaluation protocol as DiNSR:

```text
medical record -> TCM diagnosis -> syndrome -> treatment principles -> herbs
```

The primary result is the predicted chain. Oracle-chain metrics are emitted as
diagnostics. Diagnosis and syndrome use macro-F1; treatment and herbs use
micro-F1 at threshold 0.5. Checkpoints are selected by validation
predicted-chain score. A fixed `split` column is used whenever it is present.

## Available scripts

| Paper-table name | Script | Implementation |
|---|---|---|
| MacBERT | `MacBERT.py` | MacBERT encoder with dependent chain heads |
| PLE | `PLE.py` | MacBERT encoder with shared/task-specific PLE experts |
| MMoE | `MMoE.py` | MacBERT encoder with multi-gate mixture-of-experts |
| TextCNN | `TextCNN.py` | TextCNN encoder with chain heads |
| BiLSTM-Attn | `BiLSTM-Attn.py` | BiLSTM plus additive attention |
| GCN | `GCN.py` | train-split co-occurrence GCN |
| LightXML | `LightXML.py` | task-adapted label-embedding XML chain |
| GAT | `GAT.py` | train-split co-occurrence GAT |
| R-GCN | `R-GCN.py` | relation-aware co-occurrence GCN |
| HAN | `HAN.py` | heterogeneous attention network |
| HGT | `HGT.py` | heterogeneous graph transformer |
| Lexicon-Transformer | `Lexicon-Transformer.py` | Transformer over lexicon-token IDs |
| BGE-M3-LR | `BGE-M3-LR.py` | BGE-M3 embeddings plus stagewise logistic regression |

SMGCN, PreSRecST, SDPR, LH-Mix, HylLR, RHC, and ML-DELight are not included
because no verified implementation for them exists in this repository.

## Setup and data

From `DiNSR_Benchmark`, run `pip install -r requirements.txt`. The public
repository does not redistribute `data/tcm_benchmark.csv`; an authorized user
must first run `prepare_release_data.py` with the approved source and index
paths, as described in the main README.

## Run one model

Every table-named Python file is a direct entry point:

```bash
python baseline/MacBERT.py --allow_model_download --use_amp --pin_memory
python baseline/TextCNN.py --allow_model_download --epochs 50 --batch_size 16
python baseline/GCN.py --epochs 50 --batch_size 16
python baseline/BGE-M3-LR.py --allow_model_download
```

Local pretrained weights can be selected with `--bert_model_name PATH` or
`--bge_model_name PATH`. Use `--data_path PATH` to override the dataset and
`--output_dir PATH` to choose the result directory. `--max_rows 128 --epochs 1`
is intended only for smoke tests; truncation uses a seeded 75/12.5/12.5 split.

Several implementations can also be launched together:

```bash
python -m baseline.train --model_list MacBERTChain,TextCNNChain,GCNChain --allow_model_download
```

Outputs contain per-model checkpoints and logs, predicted/oracle metrics,
`all_model_results.csv`, `all_model_results.json`, and `final_report.md`.

## Reproducibility notes

- Graph statistics are built from the training split only.
- Baselines do not load DiNSR symbolic rules.
- Default seed is 42 and evaluation follows `BENCHMARK_PROTOCOL.txt`.
- `LightXML.py` is a four-stage task adaptation, not a paper-exact reproduction
  of the original extreme multi-label recipe; report it as adapted LightXML.
- MacBERT, PLE, MMoE, and BGE-M3-LR need pretrained weights. TextCNN and
  BiLSTM-Attn use a randomly initialized encoder but still need a compatible
  tokenizer; use `--allow_model_download` when it is not cached locally.
