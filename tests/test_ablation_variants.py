from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ablation_variants"))

from dinsr import LLMAuditedRuleIndex, MultiGranularityRuleIndex
from variants import (
    ABLATION_SPECS,
    PAPER_VARIANT_NAMES,
    AblationDiNSR,
    get_ablation_spec,
    transform_rule_index,
)


def sparse_matrix(rows: list[int], cols: list[int], values: list[float], shape: tuple[int, int]):
    indices = torch.tensor([rows, cols], dtype=torch.long)
    return torch.sparse_coo_tensor(indices, torch.tensor(values), shape).coalesce()


def rule_tensors(targets: list[int]) -> dict[str, torch.Tensor]:
    count = len(targets)
    return {
        "antecedent_type_codes": torch.zeros((count, 1), dtype=torch.long),
        "antecedent_indices": torch.arange(count, dtype=torch.long).unsqueeze(1),
        "antecedent_mask": torch.ones((count, 1), dtype=torch.bool),
        "target_indices": torch.tensor(targets, dtype=torch.long),
        "weights": torch.ones(count, dtype=torch.float32),
    }


def make_rule_index() -> MultiGranularityRuleIndex:
    first = LLMAuditedRuleIndex(
        matrices={
            ("positive", "token", "syndrome"): sparse_matrix([0, 1, 2], [0, 1, 2], [1.0, 2.0, 3.0], (3, 3)),
            ("negative", "token", "syndrome"): sparse_matrix([0], [2], [0.5], (3, 3)),
        },
        rule_tensors={
            ("syndrome", "positive"): rule_tensors([0, 1, 2]),
            ("syndrome", "negative"): rule_tensors([0]),
        },
        relation_keys=[
            ("positive", "token", "syndrome"),
            ("negative", "token", "syndrome"),
        ],
        source_sizes={"token": 3},
        target_sizes={"syndrome": 3, "treatment": 3, "herb": 3},
        rule_stats={"rules_kept_positive": 3, "rules_kept_negative": 1},
    )
    high = {
        ("syndrome", "positive"): rule_tensors([0, 1, 2]),
        ("syndrome", "negative"): rule_tensors([0]),
    }
    path = {
        ("herb", "positive"): rule_tensors([0, 1, 2]),
        ("herb", "negative"): rule_tensors([1]),
    }
    co = {
        "treatment": sparse_matrix([0, 1, 2], [1, 2, 0], [1.0, 1.0, 1.0], (3, 3)),
        "herb": sparse_matrix([0, 1, 2], [2, 0, 1], [1.0, 1.0, 1.0], (3, 3)),
    }
    return MultiGranularityRuleIndex(
        llm_index=first,
        high_order_rules=high,
        path_rules=path,
        cooccurrence_matrices=co,
        rule_stats={
            "rules_kept_negative": 1,
            "llm_first_order_rule_count": 4,
            "llm_multi_antecedent_rule_count": 4,
            "high_order_rule_count": 4,
            "path_rule_count": 4,
            "treatment_pair_count": 3,
            "herb_pair_count": 3,
            "llm_multi_antecedent_stats": {"rules_kept_positive": 3, "rules_kept_negative": 1},
            "high_order_stats": {"rules_kept_positive": 0, "rules_kept_negative": 0},
            "path_stats": {"rules_kept_positive": 3, "rules_kept_negative": 1},
        },
    )


class FakeNeuralChain(nn.Module):
    def __init__(self, args, mappings):
        super().__init__()
        self.num_tcm_diag = len(mappings["tcm_diag_map"])
        self.num_syndrome = len(mappings["syndrome_map"])
        self.num_treatment = len(mappings["treatment_map"])
        self.num_herb = len(mappings["herb_map"])
        self.diag_mlp = nn.Linear(4, 3)
        self.diag_head = nn.Linear(3, self.num_tcm_diag)
        self.tcm_diag_embedding = nn.Embedding(self.num_tcm_diag, 2)
        self.syndrome_mlp = nn.Linear(4 + 3 + 2, 3)
        self.syndrome_head = nn.Linear(3, self.num_syndrome)
        self.syndrome_embedding = nn.Embedding(self.num_syndrome, 2)
        self.treatment_mlp = nn.Linear(4 + 3 + 3 + 2 + 2, 3)
        self.treatment_head = nn.Linear(3, self.num_treatment)
        self.treatment_projection = nn.Linear(self.num_treatment, 2, bias=False)
        self.herb_mlp = nn.Linear(4 + 3 + 3 + 3 + 2 + 2 + 2, 3)
        self.herb_head = nn.Linear(3, self.num_herb)

    def get_base_representation(self, batch):
        return batch["base_repr"]

    @staticmethod
    def get_chain_embedding(logits, embedding_layer):
        return torch.softmax(logits, dim=1) @ embedding_layer.weight


def model_args() -> SimpleNamespace:
    return SimpleNamespace(
        rule_dropout=0.0,
        fusion_gamma_diag=0.05,
        fusion_gamma_syndrome=0.10,
        fusion_gamma_treatment=0.35,
        fusion_gamma_herb=0.35,
        fusion_gate_hidden_dim=8,
        detach_cooccurrence_probs=True,
        high_order_activation="soft_and",
        rule_source_attention=True,
        rule_source_attention_hidden_dim=8,
        cooccurrence_weight_scale=0.5,
    )


def model_mappings() -> dict:
    return {
        "western_diag_map": {0: 0, 1: 1},
        "tcm_diag_map": {0: 0, 1: 1, 2: 2},
        "syndrome_map": {0: 0, 1: 1, 2: 2},
        "treatment_map": {0: 0, 1: 1, 2: 2},
        "herb_map": {0: 0, 1: 1, 2: 2},
    }


def empty_model_rule_index() -> LLMAuditedRuleIndex:
    return LLMAuditedRuleIndex(
        source_sizes={"token": 2, "init_diag": 2},
        target_sizes={"tcm_diag": 3, "syndrome": 3, "treatment": 3, "herb": 3},
    )


def model_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.ones((2, 4), dtype=torch.long),
        "base_repr": torch.randn(2, 4),
        "western_diag": torch.tensor([0, 1]),
        "init_diag": torch.tensor([0, 1]),
        "token_ids": torch.tensor([[1, 2], [2, 0]]),
        "token_mask": torch.tensor([[True, True], [True, False]]),
        "tcm_diag": torch.tensor([0, 1]),
        "syndrome": torch.tensor([1, 2]),
        "treatment": torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
        "herb": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]),
    }


class AblationVariantTests(unittest.TestCase):
    def transform(self, name: str) -> MultiGranularityRuleIndex:
        args = SimpleNamespace(ablation_variant=name, seed=42)
        return transform_rule_index(make_rule_index(), args)

    def test_paper_names_and_configs_match(self) -> None:
        expected = {
            "OnlyN",
            "OnlyS",
            "w/o DiscrepancyGate",
            "w/o FRules",
            "w/o HRules",
            "w/o CRules",
            "w/o RandomR",
            "w/o NegRules",
        }
        self.assertEqual(set(PAPER_VARIANT_NAMES), expected)
        config_dir = ROOT / "ablation_variants" / "configs"
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in config_dir.glob("*.json")]
        self.assertEqual({payload["paper_name"] for payload in payloads}, expected)
        self.assertEqual(len(payloads), len(ABLATION_SPECS))

    def test_rule_source_removals_are_isolated(self) -> None:
        no_first = self.transform("w/o FRules")
        self.assertEqual(no_first.llm_index.relation_keys, [])
        self.assertTrue(no_first.high_order_rules)
        self.assertTrue(no_first.path_rules)
        self.assertTrue(no_first.cooccurrence_matrices)

        no_high = self.transform("w/o HRules")
        self.assertEqual(no_high.high_order_rules, {})
        self.assertTrue(no_high.llm_index.relation_keys)
        self.assertTrue(no_high.path_rules)
        self.assertTrue(no_high.cooccurrence_matrices)

        no_co = self.transform("w/o CRules")
        self.assertEqual(no_co.cooccurrence_matrices, {})
        self.assertTrue(no_co.llm_index.relation_keys)
        self.assertTrue(no_co.high_order_rules)
        self.assertTrue(no_co.path_rules)

    def test_negative_rules_are_removed_before_reasoning(self) -> None:
        transformed = self.transform("w/o NegRules")
        self.assertTrue(all(key[0] == "positive" for key in transformed.llm_index.relation_keys))
        self.assertTrue(all(key[1] == "positive" for key in transformed.llm_index.rule_tensors))
        self.assertTrue(all(key[1] == "positive" for key in transformed.high_order_rules))
        self.assertTrue(all(key[1] == "positive" for key in transformed.path_rules))
        self.assertEqual(transformed.rule_stats["rules_kept_negative"], 0)

    def test_randomized_conclusions_are_deterministic_derangements(self) -> None:
        first = self.transform("w/o RandomR")
        second = self.transform("w/o RandomR")
        original = make_rule_index()

        for collection_name in ("high_order_rules", "path_rules"):
            original_collection = getattr(original, collection_name)
            first_collection = getattr(first, collection_name)
            second_collection = getattr(second, collection_name)
            for key, tensors in original_collection.items():
                old_targets = tensors["target_indices"]
                new_targets = first_collection[key]["target_indices"]
                self.assertTrue(torch.all(old_targets != new_targets))
                self.assertTrue(torch.equal(new_targets, second_collection[key]["target_indices"]))
                self.assertTrue(torch.equal(tensors["weights"], first_collection[key]["weights"]))

        self.assertEqual(first.rule_stats["randomized_conclusion_seed"], 42)
        self.assertEqual(first.rule_stats["high_order_rule_count"], original.rule_stats["high_order_rule_count"])
        self.assertEqual(first.rule_stats["path_rule_count"], original.rule_stats["path_rule_count"])

    def test_new_ablation_files_are_ascii_english(self) -> None:
        root = ROOT / "ablation_variants"
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".py", ".md", ".json"}:
                text = path.read_text(encoding="utf-8")
                self.assertTrue(text.isascii(), path)

    def test_only_n_uses_neural_logits_throughout_predicted_chain(self) -> None:
        with patch("dinsr.TCMChainBaselineModel", FakeNeuralChain):
            model = AblationDiNSR(
                model_args(),
                model_mappings(),
                {0: 0, 1: 1},
                empty_model_rule_index(),
                get_ablation_spec("OnlyN"),
            )
        outputs = model(model_batch(), mode="predict")
        for task in ("diag", "syndrome", "treatment", "herb"):
            self.assertTrue(torch.equal(outputs["fused"][task], outputs["neural"][task]))
            self.assertTrue(torch.equal(outputs["gates"][task], torch.zeros_like(outputs["gates"][task])))

    def test_no_discrepancy_gate_uses_ungated_symbolic_injection(self) -> None:
        with patch("dinsr.TCMChainBaselineModel", FakeNeuralChain):
            model = AblationDiNSR(
                model_args(),
                model_mappings(),
                {0: 0, 1: 1},
                empty_model_rule_index(),
                get_ablation_spec("w/o DiscrepancyGate"),
            )
        neural = torch.randn(2, 3)
        symbolic = torch.randn(2, 3)
        evidence = torch.ones(2, 3)
        fused, gate, _ = model._fuse("diag", neural, symbolic, evidence, torch.zeros_like(evidence))
        gamma = torch.nn.functional.softplus(model.fusion_gamma_raw["diag"])
        expected = neural + gamma * model.symbolic_norms["diag"](symbolic)
        self.assertTrue(torch.allclose(fused, expected))
        self.assertTrue(torch.equal(gate, torch.ones_like(gate)))


if __name__ == "__main__":
    unittest.main()
