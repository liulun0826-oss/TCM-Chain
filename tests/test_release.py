from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class ReleaseTests(unittest.TestCase):
    def test_table_named_baseline_entry_points_are_included(self) -> None:
        baseline_dir = ROOT / "baseline"
        expected = {
            "MacBERT.py": "MacBERTChain",
            "PLE.py": "PLEChain",
            "MMoE.py": "MMoEChain",
            "TextCNN.py": "TextCNNChain",
            "BiLSTM-Attn.py": "BiLSTMAttentionChain",
            "GCN.py": "GCNChain",
            "LightXML.py": "LightXMLChain",
            "GAT.py": "GATChain",
            "R-GCN.py": "RGCNChain",
            "HAN.py": "HANChain",
            "HGT.py": "HGTChain",
            "Lexicon-Transformer.py": "LexiconTransformerChain",
            "BGE-M3-LR.py": "BGEM3LRChain",
        }
        self.assertTrue((baseline_dir / "README.md").is_file())
        self.assertEqual({path.name for path in baseline_dir.glob("*.py") if path.name not in {
            "__init__.py", "launcher.py", "models.py", "train.py", "utils.py"
        }}, set(expected))

        for filename, model_name in expected.items():
            tree = ast.parse((baseline_dir / filename).read_text(encoding="utf-8"))
            calls = [
                node for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run"
            ]
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0].args[0].value, model_name)

    def test_only_documented_baselines_are_registered(self) -> None:
        sys.path.insert(0, str(ROOT))
        from baseline.models import BASELINE_SPECS

        expected = {
            "MacBERTChain", "PLEChain", "MMoEChain", "TextCNNChain",
            "BiLSTMAttentionChain", "GCNChain", "LightXMLChain", "GATChain",
            "RGCNChain", "HANChain", "HGTChain", "LexiconTransformerChain",
            "BGEM3LRChain",
        }
        self.assertEqual(set(BASELINE_SPECS), expected)
        self.assertEqual({spec.name for spec in BASELINE_SPECS.values()}, expected)
        runner_counts = {runner: 0 for runner in ("torch", "torch_graph", "bge")}
        for spec in BASELINE_SPECS.values():
            runner_counts[spec.runner] += 1
        self.assertEqual(runner_counts, {"torch": 7, "torch_graph": 5, "bge": 1})

    def test_only_published_model_is_registered(self) -> None:
        from dinsr import MODEL_VARIANTS

        self.assertEqual(MODEL_VARIANTS, {"DiNSR": "dinsr"})

    def test_public_release_does_not_include_restricted_clinical_data(self) -> None:
        self.assertFalse((ROOT / "data" / "tcm_benchmark.csv").exists())
        self.assertFalse((ROOT / "data" / "deidentification_report.json").exists())
        self.assertTrue((ROOT / "data" / "clinical_data_access_notes.txt").is_file())

    def test_index_files_are_included(self) -> None:
        index_dir = ROOT / "data" / "index_data"
        expected_files = {
            "initial_diagnosis_index.csv",
            "tcm_diagnosis_index.csv",
            "syndrome_index.csv",
            "treatment_principle_index.csv",
            "herb_index.csv",
            "medical_text_lexicon_index.csv",
        }
        self.assertEqual({path.name for path in index_dir.glob("*.csv")}, expected_files)
        for filename in expected_files:
            frame = pd.read_csv(index_dir / filename, encoding="utf-8-sig", nrows=5)
            self.assertEqual(list(frame.columns), ["id", "name", "count"])

    def test_no_experiment_artifact_extensions(self) -> None:
        forbidden = {".pdf", ".pt", ".pth", ".ckpt"}
        found = [path for path in ROOT.rglob("*") if path.is_file() and path.suffix.lower() in forbidden]
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
