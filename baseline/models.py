from __future__ import annotations

import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


TASKS = ("diag", "syndrome", "treatment", "herb")
NODE_TYPES = ("token", "western_diag", "diag", "syndrome", "treatment", "herb")


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    setting: str
    runner: str
    paper: str
    description: str


def _spec(name: str, setting: str, runner: str, paper: str, description: str) -> BaselineSpec:
    return BaselineSpec(name=name, setting=setting, runner=runner, paper=paper, description=description)


BASELINE_SPECS = OrderedDict(
    (
        (
            "TextCNNChain",
            _spec("TextCNNChain", "A", "torch", "Kim 2014", "TextCNN encoder with chain heads."),
        ),
        (
            "BiLSTMAttentionChain",
            _spec(
                "BiLSTMAttentionChain",
                "A",
                "torch",
                "Yang et al. 2016",
                "BiLSTM additive-attention encoder with chain heads.",
            ),
        ),
        (
            "MacBERTChain",
            _spec("MacBERTChain", "A", "torch", "Cui et al. 2020", "MacBERT chain reference baseline."),
        ),
        (
            "MMoEChain",
            _spec("MMoEChain", "A", "torch", "Ma et al. 2018", "MacBERT text encoder with MMoE task mixing."),
        ),
        (
            "PLEChain",
            _spec("PLEChain", "A", "torch", "Tang et al. 2020", "MacBERT text encoder with PLE task mixing."),
        ),
        (
            "LexiconTransformerChain",
            _spec(
                "LexiconTransformerChain",
                "B",
                "torch",
                "Vaswani et al. 2017",
                "Transformer encoder over extracted lexicon ids.",
            ),
        ),
        (
            "LightXMLChain",
            _spec(
                "LightXMLChain",
                "B",
                "torch",
                "Jiang et al. 2021",
                "Lexicon Transformer chain with label-embedding XML heads.",
            ),
        ),
        (
            "BGEM3LRChain",
            _spec(
                "BGEM3LRChain",
                "B",
                "bge",
                "Chen et al. 2024",
                "BGE-M3 lexicon embedding followed by LR chain.",
            ),
        ),
        ("GCNChain", _spec("GCNChain", "C", "torch_graph", "Kipf and Welling 2017", "Train-split cooccurrence GCN chain.")),
        ("GATChain", _spec("GATChain", "C", "torch_graph", "Velickovic et al. 2018", "Train-split cooccurrence GAT chain.")),
        (
            "RGCNChain",
            _spec("RGCNChain", "C", "torch_graph", "Schlichtkrull et al. 2018", "Train-split cooccurrence RGCN chain."),
        ),
        ("HANChain", _spec("HANChain", "C", "torch_graph", "Wang et al. 2019", "Train-split heterogeneous attention chain.")),
        ("HGTChain", _spec("HGTChain", "C", "torch_graph", "Hu et al. 2020", "Train-split heterogeneous transformer chain.")),
    )
)


@dataclass
class GraphRelationData:
    name: str
    source_type: str
    target_type: str
    source_index: torch.Tensor
    target_index: torch.Tensor
    weight: torch.Tensor


@dataclass
class CooccurrenceGraphData:
    node_sizes: dict[str, int]
    relations: list[GraphRelationData]
    stats: dict[str, Any]


class LabelEmbeddingHead(nn.Module):
    def __init__(self, hidden_dim: int, num_labels: int, label_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, label_dim)
        self.labels = nn.Parameter(torch.empty(num_labels, label_dim))
        self.bias = nn.Parameter(torch.zeros(num_labels))
        nn.init.xavier_uniform_(self.labels)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.query(hidden) @ self.labels.t() + self.bias


class IdentityTaskMixer(nn.Module):
    def forward(self, base: torch.Tensor, task: str) -> torch.Tensor:
        return base


class MMoETaskMixer(nn.Module):
    def __init__(self, dim: int, num_experts: int, dropout: float):
        super().__init__()
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout)) for _ in range(num_experts)]
        )
        self.gates = nn.ModuleDict({task: nn.Linear(dim, num_experts) for task in TASKS})

    def forward(self, base: torch.Tensor, task: str) -> torch.Tensor:
        expert_stack = torch.stack([expert(base) for expert in self.experts], dim=1)
        gate = torch.softmax(self.gates[task](base), dim=1).unsqueeze(-1)
        return (expert_stack * gate).sum(dim=1)


class PLETaskMixer(nn.Module):
    def __init__(self, dim: int, num_shared: int, num_task: int, dropout: float):
        super().__init__()
        self.shared = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout)) for _ in range(num_shared)]
        )
        self.task_experts = nn.ModuleDict(
            {
                task: nn.ModuleList(
                    [nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout)) for _ in range(num_task)]
                )
                for task in TASKS
            }
        )
        self.gates = nn.ModuleDict({task: nn.Linear(dim, num_shared + num_task) for task in TASKS})

    def forward(self, base: torch.Tensor, task: str) -> torch.Tensor:
        experts = [expert(base) for expert in self.shared]
        experts.extend(expert(base) for expert in self.task_experts[task])
        expert_stack = torch.stack(experts, dim=1)
        gate = torch.softmax(self.gates[task](base), dim=1).unsqueeze(-1)
        return (expert_stack * gate).sum(dim=1)


class ChainPredictionModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        base_dim: int,
        mappings: dict[str, Any],
        args: Any,
        task_mixer: nn.Module | None = None,
        use_xml_heads: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.args = args
        self.num_tcm_diag = len(mappings["tcm_diag_map"])
        self.num_syndrome = len(mappings["syndrome_map"])
        self.num_treatment = len(mappings["treatment_map"])
        self.num_herb = len(mappings["herb_map"])
        hidden_dim = int(args.hidden_dim)
        embedding_dim = int(args.embedding_dim)
        dropout = float(args.dropout)

        self.task_mixer = task_mixer or IdentityTaskMixer()
        self.tcm_diag_embedding = nn.Embedding(self.num_tcm_diag, embedding_dim)
        self.syndrome_embedding = nn.Embedding(self.num_syndrome, embedding_dim)
        self.treatment_projection = nn.Linear(self.num_treatment, embedding_dim)

        self.diag_mlp = self._mlp(base_dim, hidden_dim, dropout)
        self.diag_head = nn.Linear(hidden_dim, self.num_tcm_diag)
        self.syndrome_mlp = self._mlp(base_dim + hidden_dim + embedding_dim, hidden_dim, dropout)
        self.syndrome_head = nn.Linear(hidden_dim, self.num_syndrome)
        self.treatment_mlp = self._mlp(base_dim + hidden_dim * 2 + embedding_dim * 2, hidden_dim, dropout)
        self.treatment_head = (
            LabelEmbeddingHead(hidden_dim, self.num_treatment, embedding_dim) if use_xml_heads else nn.Linear(hidden_dim, self.num_treatment)
        )
        self.herb_mlp = self._mlp(base_dim + hidden_dim * 3 + embedding_dim * 3, hidden_dim, dropout)
        self.herb_head = LabelEmbeddingHead(hidden_dim, self.num_herb, embedding_dim) if use_xml_heads else nn.Linear(hidden_dim, self.num_herb)

    @staticmethod
    def _mlp(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout))

    def _multiclass_chain_embedding(self, logits: torch.Tensor, embedding: nn.Embedding) -> torch.Tensor:
        if getattr(self.args, "chain_inference", "soft") == "hard":
            return embedding(torch.argmax(logits, dim=1))
        return torch.softmax(logits, dim=1) @ embedding.weight

    def _multilabel_chain_representation(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        if getattr(self.args, "chain_inference", "soft") == "hard":
            probs = (probs >= float(getattr(self.args, "multilabel_threshold", 0.5))).float()
        return self.treatment_projection(probs)

    def _base_for(self, encoded: torch.Tensor, task: str) -> torch.Tensor:
        return self.task_mixer(encoded, task)

    def forward(self, batch: dict[str, torch.Tensor], mode: str = "train"):
        if mode not in {"train", "oracle", "predict"}:
            raise ValueError(f"Unsupported chain mode: {mode}")
        encoded = self.encoder(batch)
        diag_base = self._base_for(encoded, "diag")
        diag_hidden = self.diag_mlp(diag_base)
        diag_logits = self.diag_head(diag_hidden)

        if mode in {"train", "oracle"}:
            diag_repr = self.tcm_diag_embedding(batch["tcm_diag"])
        else:
            diag_repr = self._multiclass_chain_embedding(diag_logits, self.tcm_diag_embedding)

        syndrome_base = self._base_for(encoded, "syndrome")
        syndrome_hidden = self.syndrome_mlp(torch.cat([syndrome_base, diag_hidden, diag_repr], dim=1))
        syndrome_logits = self.syndrome_head(syndrome_hidden)

        if mode in {"train", "oracle"}:
            syndrome_repr = self.syndrome_embedding(batch["syndrome"])
        else:
            syndrome_repr = self._multiclass_chain_embedding(syndrome_logits, self.syndrome_embedding)

        treatment_base = self._base_for(encoded, "treatment")
        treatment_hidden = self.treatment_mlp(
            torch.cat([treatment_base, diag_hidden, syndrome_hidden, diag_repr, syndrome_repr], dim=1)
        )
        treatment_logits = self.treatment_head(treatment_hidden)

        if mode in {"train", "oracle"}:
            treatment_repr = self.treatment_projection(batch["treatment"])
        else:
            treatment_repr = self._multilabel_chain_representation(treatment_logits)

        herb_base = self._base_for(encoded, "herb")
        herb_hidden = self.herb_mlp(
            torch.cat(
                [
                    herb_base,
                    diag_hidden,
                    syndrome_hidden,
                    treatment_hidden,
                    diag_repr,
                    syndrome_repr,
                    treatment_repr,
                ],
                dim=1,
            )
        )
        herb_logits = self.herb_head(herb_hidden)
        return diag_logits, syndrome_logits, treatment_logits, herb_logits


class StructuredFusionEncoder(nn.Module):
    def __init__(self, representation_dim: int, mappings: dict[str, Any], args: Any):
        super().__init__()
        embedding_dim = int(args.embedding_dim)
        self.age_mlp = nn.Linear(1, embedding_dim)
        self.sex_embedding = nn.Embedding(2, embedding_dim)
        self.western_diag_embedding = nn.Embedding(len(mappings["western_diag_map"]), embedding_dim)
        self.output_dim = representation_dim + embedding_dim * 3

    def fuse(self, representation: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        age_repr = self.age_mlp(batch["age"].unsqueeze(1))
        sex_repr = self.sex_embedding(batch["sex"])
        western_repr = self.western_diag_embedding(batch["western_diag"])
        return torch.cat([representation, age_repr, sex_repr, western_repr], dim=1)


class HFTextEncoder(StructuredFusionEncoder):
    def __init__(self, mappings: dict[str, Any], args: Any):
        try:
            from transformers import AutoModel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("MacBERT/MMoE/PLE baselines require `transformers`.") from exc
        transformer = AutoModel.from_pretrained(
            args.bert_model_name,
            local_files_only=getattr(args, "local_files_only", False),
        )
        super().__init__(transformer.config.hidden_size, mappings, args)
        self.transformer = transformer
        if getattr(args, "freeze_bert", False):
            for param in self.transformer.parameters():
                param.requires_grad = False

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.transformer(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        return self.fuse(output.last_hidden_state[:, 0, :], batch)


class TextCNNEncoder(StructuredFusionEncoder):
    def __init__(self, tokenizer_vocab_size: int, mappings: dict[str, Any], args: Any):
        representation_dim = int(args.text_encoder_dim)
        super().__init__(representation_dim, mappings, args)
        kernel_sizes = tuple(int(size) for size in str(args.textcnn_kernel_sizes).split(",") if size.strip())
        channels = max(8, representation_dim // max(len(kernel_sizes), 1))
        self.embedding = nn.Embedding(tokenizer_vocab_size, int(args.embedding_dim), padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(int(args.embedding_dim), channels, kernel_size=size) for size in kernel_sizes])
        self.projection = nn.Linear(channels * len(kernel_sizes), representation_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        embedded = self.embedding(batch["input_ids"]).transpose(1, 2)
        pooled = [F.adaptive_max_pool1d(F.relu(conv(embedded)), 1).squeeze(-1) for conv in self.convs]
        return self.fuse(self.projection(torch.cat(pooled, dim=1)), batch)


class BiLSTMAttentionEncoder(StructuredFusionEncoder):
    def __init__(self, tokenizer_vocab_size: int, mappings: dict[str, Any], args: Any):
        representation_dim = int(args.text_encoder_dim)
        super().__init__(representation_dim, mappings, args)
        lstm_hidden = max(8, representation_dim // 2)
        self.embedding = nn.Embedding(tokenizer_vocab_size, int(args.embedding_dim), padding_idx=0)
        self.lstm = nn.LSTM(int(args.embedding_dim), lstm_hidden, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(lstm_hidden * 2, 1)
        self.projection = nn.Linear(lstm_hidden * 2, representation_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        states, _ = self.lstm(self.embedding(batch["input_ids"]))
        scores = self.attention(torch.tanh(states)).squeeze(-1)
        scores = scores.masked_fill(~batch["attention_mask"].bool(), -1e4)
        attention = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (states * attention).sum(dim=1)
        return self.fuse(self.projection(pooled), batch)


class LexiconTransformerEncoder(StructuredFusionEncoder):
    def __init__(self, token_map: dict[int, int], mappings: dict[str, Any], args: Any):
        representation_dim = int(args.lexicon_encoder_dim)
        super().__init__(representation_dim, mappings, args)
        self.embedding = nn.Embedding(len(token_map) + 1, representation_dim, padding_idx=0)
        layer = nn.TransformerEncoderLayer(
            d_model=representation_dim,
            nhead=int(args.lexicon_num_heads),
            dim_feedforward=representation_dim * 2,
            dropout=float(args.dropout),
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(args.lexicon_num_layers))
        self.norm = nn.LayerNorm(representation_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = batch["token_mask"].bool()
        states = self.transformer(self.embedding(batch["token_ids"]), src_key_padding_mask=~mask)
        pooled = (states * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return self.fuse(self.norm(pooled), batch)


def _safe_module_key(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]", "_", value)


class HeteroCooccurrenceEncoder(nn.Module):
    def __init__(self, graph_data: CooccurrenceGraphData, args: Any, variant: str):
        super().__init__()
        self.variant = variant
        self.relations = graph_data.relations
        graph_dim = int(args.graph_dim)
        embedding_dim = int(args.embedding_dim)
        self.node_embeddings = nn.ModuleDict(
            {node_type: nn.Embedding(max(1, int(graph_data.node_sizes[node_type])), graph_dim) for node_type in NODE_TYPES}
        )
        self.self_linears = nn.ModuleDict({node_type: nn.Linear(graph_dim, graph_dim) for node_type in NODE_TYPES})
        self.age_mlp = nn.Linear(1, embedding_dim)
        self.sex_embedding = nn.Embedding(2, embedding_dim)
        self.output_dim = graph_dim * 2 + embedding_dim * 2
        relation_keys = [_safe_module_key(relation.name) for relation in self.relations]
        self.shared_message_linear = nn.Linear(graph_dim, graph_dim, bias=False) if variant == "gcn" else None
        self.relation_linears = nn.ModuleDict(
            {} if variant == "gcn" else {key: nn.Linear(graph_dim, graph_dim, bias=False) for key in relation_keys}
        )
        self.semantic_scores = nn.ParameterDict(
            {key: nn.Parameter(torch.zeros(())) for key in relation_keys} if variant == "han" else {}
        )
        self.type_q = nn.ModuleDict(
            {node_type: nn.Linear(graph_dim, graph_dim, bias=False) for node_type in NODE_TYPES}
            if variant == "hgt" else {}
        )
        self.type_k = nn.ModuleDict(
            {node_type: nn.Linear(graph_dim, graph_dim, bias=False) for node_type in NODE_TYPES}
            if variant == "hgt" else {}
        )
        self.type_v = nn.ModuleDict(
            {node_type: nn.Linear(graph_dim, graph_dim, bias=False) for node_type in NODE_TYPES}
            if variant == "hgt" else {}
        )
        self.norms = nn.ModuleDict({node_type: nn.LayerNorm(graph_dim) for node_type in NODE_TYPES})
        self.dropout = nn.Dropout(float(args.dropout))

        for index, relation in enumerate(self.relations):
            prefix = f"relation_{index}"
            self.register_buffer(f"{prefix}_source", relation.source_index.long(), persistent=False)
            self.register_buffer(f"{prefix}_target", relation.target_index.long(), persistent=False)
            self.register_buffer(f"{prefix}_weight", relation.weight.float(), persistent=False)

    def _relation_buffers(self, index: int):
        return (
            getattr(self, f"relation_{index}_source"),
            getattr(self, f"relation_{index}_target"),
            getattr(self, f"relation_{index}_weight"),
        )

    def _message_for_relation(
        self,
        index: int,
        relation: GraphRelationData,
        base: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        source_index, target_index, weight = self._relation_buffers(index)
        source_states = base[relation.source_type][source_index]
        target_states = base[relation.target_type][target_index]
        key = _safe_module_key(relation.name)

        if self.variant == "gcn":
            message = self.shared_message_linear(source_states)
            edge_weight = weight
        elif self.variant == "gat":
            source_proj = self.relation_linears[key](source_states)
            target_proj = self.self_linears[relation.target_type](target_states)
            edge_weight = weight * torch.sigmoid((source_proj * target_proj).sum(dim=1) / math.sqrt(source_proj.shape[1]))
            message = source_proj
        elif self.variant == "hgt":
            query = self.type_q[relation.target_type](target_states)
            key_state = self.type_k[relation.source_type](source_states)
            value = self.type_v[relation.source_type](source_states)
            edge_weight = weight * torch.sigmoid((query * key_state).sum(dim=1) / math.sqrt(query.shape[1]))
            message = self.relation_linears[key](value)
        else:
            message = self.relation_linears[key](source_states)
            edge_weight = weight

        aggregated = torch.zeros_like(base[relation.target_type])
        normalizer = torch.zeros(base[relation.target_type].shape[0], 1, device=aggregated.device)
        aggregated.index_add_(0, target_index, message * edge_weight.unsqueeze(1))
        normalizer.index_add_(0, target_index, edge_weight.unsqueeze(1))
        return aggregated / normalizer.clamp(min=1e-6)

    def _propagate(self) -> dict[str, torch.Tensor]:
        base = {node_type: embedding.weight for node_type, embedding in self.node_embeddings.items()}
        relation_messages = {node_type: [] for node_type in NODE_TYPES}
        relation_keys = {node_type: [] for node_type in NODE_TYPES}
        for index, relation in enumerate(self.relations):
            if relation.source_index.numel() == 0:
                continue
            relation_messages[relation.target_type].append(self._message_for_relation(index, relation, base))
            relation_keys[relation.target_type].append(_safe_module_key(relation.name))

        propagated = {}
        for node_type in NODE_TYPES:
            if not relation_messages[node_type]:
                update = torch.zeros_like(base[node_type])
            elif self.variant == "han":
                semantic = torch.softmax(
                    torch.stack([self.semantic_scores[key] for key in relation_keys[node_type]]),
                    dim=0,
                )
                update = sum(weight * message for weight, message in zip(semantic, relation_messages[node_type]))
            else:
                update = torch.stack(relation_messages[node_type], dim=0).mean(dim=0)
            propagated[node_type] = self.norms[node_type](
                self.self_linears[node_type](base[node_type]) + self.dropout(F.relu(update))
            )
        return propagated

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        graph_states = self._propagate()
        token_mask = batch["token_mask"].bool()
        token_index = (batch["token_ids"] - 1).clamp(min=0)
        token_states = graph_states["token"][token_index]
        token_repr = (token_states * token_mask.unsqueeze(-1)).sum(dim=1) / token_mask.sum(dim=1, keepdim=True).clamp(min=1)
        western_repr = graph_states["western_diag"][batch["western_diag"]]
        age_repr = self.age_mlp(batch["age"].unsqueeze(1))
        sex_repr = self.sex_embedding(batch["sex"])
        return torch.cat([token_repr, western_repr, age_repr, sex_repr], dim=1)


def _task_mixer(kind: str, dim: int, args: Any) -> nn.Module:
    if kind == "mmoe":
        return MMoETaskMixer(dim, int(args.mmoe_num_experts), float(args.dropout))
    if kind == "ple":
        return PLETaskMixer(dim, int(args.ple_shared_experts), int(args.ple_task_experts), float(args.dropout))
    return IdentityTaskMixer()


def build_torch_model(
    model_name: str,
    args: Any,
    mappings: dict[str, Any],
    token_map: dict[int, int],
    tokenizer_vocab_size: int | None = None,
    graph_data: CooccurrenceGraphData | None = None,
) -> nn.Module:
    if model_name == "TextCNNChain":
        if tokenizer_vocab_size is None:
            raise ValueError("TextCNNChain requires a tokenizer vocabulary size.")
        encoder = TextCNNEncoder(tokenizer_vocab_size, mappings, args)
        return ChainPredictionModel(encoder, encoder.output_dim, mappings, args)
    if model_name == "BiLSTMAttentionChain":
        if tokenizer_vocab_size is None:
            raise ValueError("BiLSTMAttentionChain requires a tokenizer vocabulary size.")
        encoder = BiLSTMAttentionEncoder(tokenizer_vocab_size, mappings, args)
        return ChainPredictionModel(encoder, encoder.output_dim, mappings, args)
    if model_name in {"MacBERTChain", "MMoEChain", "PLEChain"}:
        encoder = HFTextEncoder(mappings, args)
        mixer_kind = {"MMoEChain": "mmoe", "PLEChain": "ple"}.get(model_name, "identity")
        return ChainPredictionModel(encoder, encoder.output_dim, mappings, args, task_mixer=_task_mixer(mixer_kind, encoder.output_dim, args))
    if model_name in {"LexiconTransformerChain", "LightXMLChain"}:
        encoder = LexiconTransformerEncoder(token_map, mappings, args)
        return ChainPredictionModel(
            encoder,
            encoder.output_dim,
            mappings,
            args,
            use_xml_heads=model_name == "LightXMLChain",
        )
    if model_name in {"GCNChain", "GATChain", "RGCNChain", "HANChain", "HGTChain"}:
        if graph_data is None:
            raise ValueError(f"{model_name} requires train-split cooccurrence graph data.")
        variant = {
            "GCNChain": "gcn",
            "GATChain": "gat",
            "RGCNChain": "rgcn",
            "HANChain": "han",
            "HGTChain": "hgt",
        }[model_name]
        encoder = HeteroCooccurrenceEncoder(graph_data, args, variant)
        return ChainPredictionModel(encoder, encoder.output_dim, mappings, args)
    raise ValueError(f"Model {model_name} is not a torch comparison model.")
