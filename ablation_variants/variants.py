from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F

from dinsr import DiNSR, LLMAuditedRuleIndex, MultiGranularityRuleIndex


@dataclass(frozen=True)
class AblationSpec:
    paper_name: str
    slug: str
    intervention: str
    scientific_control: str

    def as_dict(self) -> dict[str, str]:
        return {
            "paper_name": self.paper_name,
            "slug": self.slug,
            "intervention": self.intervention,
            "scientific_control": self.scientific_control,
        }


ABLATION_SPECS = {
    spec.paper_name: spec
    for spec in (
        AblationSpec(
            "OnlyN",
            "only_n",
            "Use the neural chain only; symbolic reasoning and fusion are bypassed.",
            "The neural architecture, optimization schedule, data split, and PC/OC evaluation remain unchanged.",
        ),
        AblationSpec(
            "OnlyS",
            "only_s",
            "Use the symbolic branch only; neural logits are excluded from every prediction.",
            "Multi-label co-occurrence evidence is seeded by a detached symbolic first pass, not by neural probabilities.",
        ),
        AblationSpec(
            "w/o DiscrepancyGate",
            "wo_discrepancy_gate",
            "Replace patient-specific discrepancy gating with ungated task-specific symbolic injection.",
            "The symbolic branch and learnable task-specific fusion coefficient are retained.",
        ),
        AblationSpec(
            "w/o FRules",
            "wo_f_rules",
            "Remove all audited first-order rules.",
            "High-order, path, and co-occurrence rules are retained without changing their weights.",
        ),
        AblationSpec(
            "w/o HRules",
            "wo_h_rules",
            "Remove all high-order multi-antecedent rules.",
            "First-order, path, and co-occurrence rules are retained without changing their weights.",
        ),
        AblationSpec(
            "w/o CRules",
            "wo_c_rules",
            "Remove treatment and herb co-occurrence rules.",
            "First-order, high-order, and path rules are retained without changing their weights.",
        ),
        AblationSpec(
            "w/o RandomR",
            "wo_random_r",
            "Replace each rule conclusion with a deterministic derangement within the same task label space.",
            "Antecedents, signs, rule counts, evidence weights, and source types are preserved.",
        ),
        AblationSpec(
            "w/o NegRules",
            "wo_neg_rules",
            "Remove all negative conflict rules before rule-source attention.",
            "All positive rules and their original evidence weights are retained.",
        ),
    )
}
PAPER_VARIANT_NAMES = tuple(ABLATION_SPECS)


def get_ablation_spec(paper_name: str) -> AblationSpec:
    try:
        return ABLATION_SPECS[paper_name]
    except KeyError as exc:
        valid = ", ".join(PAPER_VARIANT_NAMES)
        raise ValueError(f"Unknown ablation variant: {paper_name}. Valid variants: {valid}") from exc


def apply_variant_training_controls(args: Any, spec: AblationSpec) -> None:
    args.ablation_variant = spec.paper_name
    if spec.paper_name == "OnlyN":
        args.alpha_neural = 0.0
        args.alpha_symbolic = 0.0
        args.beta_rule = 0.0
        args.fusion_mode = "neural_only"
    elif spec.paper_name == "OnlyS":
        args.alpha_neural = 0.0
        args.alpha_symbolic = 0.0
        args.beta_rule = 0.0
        args.fusion_mode = "symbolic_only_two_pass_cooccurrence"
    elif spec.paper_name == "w/o DiscrepancyGate":
        args.fusion_mode = "ungated_symbolic_injection"
    else:
        args.fusion_mode = "learned_discrepancy_gate"


def _count_rules(rule_tensors: dict[tuple[str, str], dict[str, torch.Tensor]]) -> int:
    return int(sum(int(tensors["target_indices"].numel()) for tensors in rule_tensors.values()))


def _filter_positive_first_order(index: LLMAuditedRuleIndex) -> LLMAuditedRuleIndex:
    matrices = {key: value for key, value in index.matrices.items() if key[0] == "positive"}
    relation_keys = [key for key in index.relation_keys if key[0] == "positive"]
    rule_tensors = {key: value for key, value in index.rule_tensors.items() if key[1] == "positive"}
    stats = dict(index.rule_stats)
    stats["rules_kept_negative"] = 0
    stats["rules_kept"] = int(stats.get("rules_kept_positive", _count_rules(rule_tensors)))
    return replace(
        index,
        matrices=matrices,
        relation_keys=relation_keys,
        rule_tensors=rule_tensors,
        rule_stats=stats,
    )


def _empty_first_order(index: LLMAuditedRuleIndex) -> LLMAuditedRuleIndex:
    stats = dict(index.rule_stats)
    stats.update({"rules_kept": 0, "rules_kept_positive": 0, "rules_kept_negative": 0})
    return replace(index, matrices={}, relation_keys=[], rule_tensors={}, rule_stats=stats)


def _positive_rule_tensors(
    rule_tensors: dict[tuple[str, str], dict[str, torch.Tensor]],
) -> dict[tuple[str, str], dict[str, torch.Tensor]]:
    return {key: value for key, value in rule_tensors.items() if key[1] == "positive"}


def _derangement(size: int, seed: int) -> torch.Tensor:
    if size <= 1:
        return torch.arange(size, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    identity = torch.arange(size, dtype=torch.long)
    for _ in range(128):
        permutation = torch.randperm(size, generator=generator)
        if bool(torch.all(permutation != identity)):
            return permutation
    shift = int(torch.randint(1, size, (1,), generator=generator).item())
    return torch.roll(identity, shifts=shift)


def _permute_sparse_target_rows(matrix: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    matrix = matrix.coalesce()
    indices = matrix.indices().clone()
    indices[0] = permutation.to(indices.device).index_select(0, indices[0])
    return torch.sparse_coo_tensor(indices, matrix.values().clone(), matrix.shape).coalesce()


def _permute_rule_targets(
    tensors: dict[str, torch.Tensor], permutation: torch.Tensor
) -> dict[str, torch.Tensor]:
    transformed = {key: value.clone() for key, value in tensors.items()}
    target_indices = transformed.get("target_indices")
    if target_indices is not None and target_indices.numel() > 0:
        transformed["target_indices"] = permutation.to(target_indices.device).index_select(0, target_indices)
    return transformed


def _task_permutations(index: LLMAuditedRuleIndex, seed: int) -> dict[str, torch.Tensor]:
    offsets = {"tcm_diag": 101, "syndrome": 211, "treatment": 307, "herb": 401}
    return {
        target_type: _derangement(int(index.target_sizes[target_type]), seed + offset)
        for target_type, offset in offsets.items()
        if target_type in index.target_sizes
    }


def _randomize_first_order(
    index: LLMAuditedRuleIndex, permutations: dict[str, torch.Tensor]
) -> LLMAuditedRuleIndex:
    matrices = {
        key: _permute_sparse_target_rows(matrix, permutations[key[2]])
        if key[2] in permutations
        else matrix.clone()
        for key, matrix in index.matrices.items()
    }
    rule_tensors = {
        key: _permute_rule_targets(tensors, permutations[key[0]])
        if key[0] in permutations
        else {name: tensor.clone() for name, tensor in tensors.items()}
        for key, tensors in index.rule_tensors.items()
    }
    return replace(index, matrices=matrices, rule_tensors=rule_tensors)


def _randomize_multi_rules(
    rule_tensors: dict[tuple[str, str], dict[str, torch.Tensor]],
    permutations: dict[str, torch.Tensor],
) -> dict[tuple[str, str], dict[str, torch.Tensor]]:
    return {
        key: _permute_rule_targets(tensors, permutations[key[0]])
        if key[0] in permutations
        else {name: tensor.clone() for name, tensor in tensors.items()}
        for key, tensors in rule_tensors.items()
    }


def transform_rule_index(
    rule_index: LLMAuditedRuleIndex | MultiGranularityRuleIndex,
    args: Any,
) -> LLMAuditedRuleIndex | MultiGranularityRuleIndex:
    spec = get_ablation_spec(args.ablation_variant)
    name = spec.paper_name

    if isinstance(rule_index, LLMAuditedRuleIndex):
        if name == "w/o FRules":
            transformed = _empty_first_order(rule_index)
        elif name == "w/o NegRules":
            transformed = _filter_positive_first_order(rule_index)
        elif name == "w/o RandomR":
            permutations = _task_permutations(rule_index, int(args.seed))
            transformed = _randomize_first_order(rule_index, permutations)
        else:
            transformed = rule_index
        transformed.rule_stats = {
            **transformed.rule_stats,
            "ablation_variant": name,
            "ablation_intervention": spec.intervention,
        }
        return transformed

    llm_index = rule_index.llm_index
    high_order_rules = rule_index.high_order_rules
    path_rules = rule_index.path_rules
    cooccurrence_matrices = rule_index.cooccurrence_matrices
    stats = dict(rule_index.rule_stats)

    if name == "w/o FRules":
        llm_index = _empty_first_order(llm_index)
        stats["llm_first_order_rule_count"] = 0
    elif name == "w/o HRules":
        high_order_rules = {}
        stats["llm_multi_antecedent_rule_count"] = 0
        stats["high_order_rule_count"] = 0
    elif name == "w/o CRules":
        cooccurrence_matrices = {}
        stats["treatment_pair_count"] = 0
        stats["herb_pair_count"] = 0
    elif name == "w/o NegRules":
        llm_index = _filter_positive_first_order(llm_index)
        high_order_rules = _positive_rule_tensors(high_order_rules)
        path_rules = _positive_rule_tensors(path_rules)
        audited_positive = int(stats.get("llm_multi_antecedent_stats", {}).get("rules_kept_positive", 0))
        high_file_positive = int(stats.get("high_order_stats", {}).get("rules_kept_positive", 0))
        path_positive = int(stats.get("path_stats", {}).get("rules_kept_positive", 0))
        stats["rules_kept_negative"] = 0
        stats["llm_first_order_rule_count"] = _count_rules(llm_index.rule_tensors)
        stats["llm_multi_antecedent_rule_count"] = audited_positive
        stats["high_order_rule_count"] = audited_positive + high_file_positive
        stats["path_rule_count"] = path_positive
    elif name == "w/o RandomR":
        permutations = _task_permutations(llm_index, int(args.seed))
        llm_index = _randomize_first_order(llm_index, permutations)
        high_order_rules = _randomize_multi_rules(high_order_rules, permutations)
        path_rules = _randomize_multi_rules(path_rules, permutations)
        cooccurrence_matrices = {
            target_type: _permute_sparse_target_rows(matrix, permutations[target_type])
            if target_type in permutations
            else matrix.clone()
            for target_type, matrix in cooccurrence_matrices.items()
        }
        stats["randomized_conclusion_seed"] = int(args.seed)
        stats["randomization_strategy"] = "task-wise deterministic derangement of consequent labels"

    stats.update(
        {
            "ablation_variant": name,
            "ablation_intervention": spec.intervention,
            "ablation_scientific_control": spec.scientific_control,
        }
    )
    return replace(
        rule_index,
        llm_index=llm_index,
        high_order_rules=high_order_rules,
        path_rules=path_rules,
        cooccurrence_matrices=cooccurrence_matrices,
        rule_stats=stats,
    )


class AblationDiNSR(DiNSR):
    def __init__(
        self,
        args: Any,
        mappings: dict[str, Any],
        token_map: dict[int, int],
        rule_index: LLMAuditedRuleIndex | MultiGranularityRuleIndex,
        spec: AblationSpec,
    ):
        self.ablation_spec = spec
        super().__init__(args, mappings, token_map, rule_index)
        if spec.paper_name == "OnlyN":
            self._freeze(self.reasoner)
            self._freeze(self.symbolic_norms)
            self._freeze(self.label_mlps)
            self._freeze(self.gate_mlps)
            self._freeze(self.fusion_gamma_raw)
        elif spec.paper_name == "OnlyS":
            self._freeze(self.neural)
            self._freeze(self.gate_mlps)
            self._freeze(self.fusion_gamma_raw)
        elif spec.paper_name == "w/o DiscrepancyGate":
            self._freeze(self.gate_mlps)

    @staticmethod
    def _freeze(module: torch.nn.Module) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    def _symbolic_component(
        self,
        task: str,
        activations: dict[str, torch.Tensor],
        base_repr: torch.Tensor,
        current_task_probs: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        name = self.ablation_spec.paper_name
        if name == "OnlyN":
            dimensions = {
                "diag": self.num_tcm_diag,
                "syndrome": self.num_syndrome,
                "treatment": self.num_treatment,
                "herb": self.num_herb,
            }
            zero = base_repr.new_zeros((base_repr.shape[0], dimensions[task]))
            return zero, zero, zero, {"ablation": "symbolic branch bypassed"}

        if name == "OnlyS" and task in {"treatment", "herb"} and self.uses_multi_granularity_rules:
            first_logits, _, _, first_details = super()._symbolic_component(
                task, activations, base_repr, current_task_probs=None
            )
            seed_probs = torch.sigmoid(self.symbolic_norms[task](first_logits))
            if bool(getattr(self.args, "detach_cooccurrence_probs", True)):
                seed_probs = seed_probs.detach()
            logits, pos, neg, details = super()._symbolic_component(
                task,
                activations,
                base_repr,
                current_task_probs={task: seed_probs},
            )
            details = dict(details)
            details["symbolic_seed_source"] = "detached first-pass symbolic probabilities"
            details["first_pass_source_weights"] = first_details.get("source_weights")
            return logits, pos, neg, details

        return super()._symbolic_component(task, activations, base_repr, current_task_probs)

    def _fuse(
        self,
        task: str,
        neural_logits: torch.Tensor,
        symbolic_logits: torch.Tensor,
        pos_logits: torch.Tensor,
        neg_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        name = self.ablation_spec.paper_name
        symbolic_component = self.symbolic_norms[task](symbolic_logits)
        disagreement = self._disagreement(task, neural_logits, symbolic_component)
        batch_size = neural_logits.shape[0]

        if name == "OnlyN":
            return neural_logits, neural_logits.new_zeros((batch_size, 1)), disagreement
        if name == "OnlyS":
            return symbolic_component, neural_logits.new_ones((batch_size, 1)), disagreement
        if name == "w/o DiscrepancyGate":
            gamma = F.softplus(self.fusion_gamma_raw[task])
            fused = neural_logits + gamma * symbolic_component
            return fused, neural_logits.new_ones((batch_size, 1)), disagreement
        return super()._fuse(task, neural_logits, symbolic_logits, pos_logits, neg_logits)


def create_ablation_model(
    args: Any,
    mappings: dict[str, Any],
    token_map: dict[int, int],
    rule_index: LLMAuditedRuleIndex | MultiGranularityRuleIndex,
    model_name: str,
) -> AblationDiNSR:
    spec = get_ablation_spec(model_name)
    return AblationDiNSR(args, mappings, token_map, rule_index, spec)
