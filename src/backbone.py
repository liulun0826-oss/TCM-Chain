import torch
import torch.nn as nn
from transformers import AutoModel

class TCMChainBaselineModel(nn.Module):
    def __init__(self, args, mappings):
        super().__init__()
        self.args = args
        self.mappings = mappings

        # 1. Text encoder
        self.bert = AutoModel.from_pretrained(
            args.bert_model_name,
            local_files_only=getattr(args, "local_files_only", False),
        )
        if args.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        bert_hidden_size = self.bert.config.hidden_size

        # 2. Structured feature encoders
        self.age_mlp = nn.Linear(1, args.embedding_dim)
        self.sex_embedding = nn.Embedding(2, args.embedding_dim)
        
        num_western_diag = len(mappings['western_diag_map'])
        self.western_diag_embedding = nn.Embedding(num_western_diag, args.embedding_dim)

        self.num_tcm_diag = len(mappings['tcm_diag_map'])
        self.tcm_diag_embedding = nn.Embedding(self.num_tcm_diag, args.embedding_dim)

        self.num_syndrome = len(mappings['syndrome_map'])
        self.syndrome_embedding = nn.Embedding(self.num_syndrome, args.embedding_dim)

        self.num_treatment = len(mappings['treatment_map'])
        self.treatment_projection = nn.Linear(self.num_treatment, args.embedding_dim)

        # 3. Fusion layers and prediction heads
        self.base_repr_dim = bert_hidden_size + args.embedding_dim * 3

        # Task 1: predict the TCM diagnosis.
        self.diag_mlp = nn.Sequential(
            nn.Linear(self.base_repr_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout)
        )
        self.diag_head = nn.Linear(args.hidden_dim, self.num_tcm_diag)

        # Task 2: predict the syndrome.
        self.syndrome_mlp = nn.Sequential(
            nn.Linear(self.base_repr_dim + args.hidden_dim + args.embedding_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout)
        )
        self.syndrome_head = nn.Linear(args.hidden_dim, self.num_syndrome)

        # Task 3: predict treatment principles.
        self.treatment_mlp = nn.Sequential(
            nn.Linear(
                self.base_repr_dim + args.hidden_dim * 2 + args.embedding_dim * 2,
                args.hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(args.dropout)
        )
        self.treatment_head = nn.Linear(args.hidden_dim, self.num_treatment)

        # Task 4: predict herbs.
        self.herb_mlp = nn.Sequential(
            nn.Linear(
                self.base_repr_dim + args.hidden_dim * 3 + args.embedding_dim * 3,
                args.hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(args.dropout)
        )
        self.num_herb = len(mappings['herb_map'])
        self.herb_head = nn.Linear(args.hidden_dim, self.num_herb)

    def forward(self, batch, mode='train'):
        # Training mode with teacher forcing, or oracle-chain evaluation.
        if mode in ['train', 'oracle']:
            return self.forward_teacher_forcing(batch)
        # Predicted-chain evaluation.
        elif mode == 'predict':
            return self.forward_predict_chain(batch)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def get_base_representation(self, batch):
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        age, sex, western_diag = batch['age'], batch['sex'], batch['western_diag']

        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_repr = bert_output.last_hidden_state[:, 0, :]  # [CLS] token

        age_repr = self.age_mlp(age.unsqueeze(1))
        sex_repr = self.sex_embedding(sex)
        western_diag_repr = self.western_diag_embedding(western_diag)

        base_repr = torch.cat([text_repr, age_repr, sex_repr, western_diag_repr], dim=1)
        return base_repr

    def get_chain_embedding(self, logits, embedding_layer):
        if getattr(self.args, "chain_inference", "soft") == "hard":
            pred = torch.argmax(logits, dim=1)
            return embedding_layer(pred)
        probs = torch.softmax(logits, dim=1)
        return probs @ embedding_layer.weight

    def forward_teacher_forcing(self, batch):
        base_repr = self.get_base_representation(batch)

        # Task 1: predict the TCM diagnosis.
        diag_hidden = self.diag_mlp(base_repr)
        diag_logits = self.diag_head(diag_hidden)

        # Task 2: predict the syndrome from base features, diagnosis state, and true TCM diagnosis.
        tcm_diag_true_emb = self.tcm_diag_embedding(batch['tcm_diag'])
        syndrome_input = torch.cat([base_repr, diag_hidden, tcm_diag_true_emb], dim=1)
        syndrome_hidden = self.syndrome_mlp(syndrome_input)
        syndrome_logits = self.syndrome_head(syndrome_hidden)

        # Task 3: predict treatment principles from base features, upstream states, and true upstream labels.
        syndrome_true_emb = self.syndrome_embedding(batch['syndrome'])
        treatment_input = torch.cat([
            base_repr,
            diag_hidden,
            syndrome_hidden,
            tcm_diag_true_emb,
            syndrome_true_emb,
        ], dim=1)
        treatment_hidden = self.treatment_mlp(treatment_input)
        treatment_logits = self.treatment_head(treatment_hidden)

        # Task 4: predict herbs from base features, all upstream states, and true upstream labels.
        treatment_true_repr = self.treatment_projection(batch['treatment'])
        herb_input = torch.cat([
            base_repr,
            diag_hidden,
            syndrome_hidden,
            treatment_hidden,
            tcm_diag_true_emb,
            syndrome_true_emb,
            treatment_true_repr,
        ], dim=1)
        herb_hidden = self.herb_mlp(herb_input)
        herb_logits = self.herb_head(herb_hidden)

        return diag_logits, syndrome_logits, treatment_logits, herb_logits

    def forward_predict_chain(self, batch):
        base_repr = self.get_base_representation(batch)

        # Task 1: predict the TCM diagnosis.
        diag_hidden = self.diag_mlp(base_repr)
        diag_logits = self.diag_head(diag_hidden)

        # Task 2: predict the syndrome from base features, diagnosis state, and predicted TCM diagnosis.
        tcm_diag_pred_emb = self.get_chain_embedding(diag_logits, self.tcm_diag_embedding)
        syndrome_input = torch.cat([base_repr, diag_hidden, tcm_diag_pred_emb], dim=1)
        syndrome_hidden = self.syndrome_mlp(syndrome_input)
        syndrome_logits = self.syndrome_head(syndrome_hidden)

        # Task 3: predict treatment principles from base features, upstream states, and predicted upstream labels.
        syndrome_pred_emb = self.get_chain_embedding(syndrome_logits, self.syndrome_embedding)
        treatment_input = torch.cat([
            base_repr,
            diag_hidden,
            syndrome_hidden,
            tcm_diag_pred_emb,
            syndrome_pred_emb,
        ], dim=1)
        treatment_hidden = self.treatment_mlp(treatment_input)
        treatment_logits = self.treatment_head(treatment_hidden)
        treatment_pred_probs = torch.sigmoid(treatment_logits)

        # Task 4: predict herbs from base features, all upstream states, and predicted upstream labels.
        treatment_pred_repr = self.treatment_projection(treatment_pred_probs)
        herb_input = torch.cat([
            base_repr,
            diag_hidden,
            syndrome_hidden,
            treatment_hidden,
            tcm_diag_pred_emb,
            syndrome_pred_emb,
            treatment_pred_repr,
        ], dim=1)
        herb_hidden = self.herb_mlp(herb_input)
        herb_logits = self.herb_head(herb_hidden)

        return diag_logits, syndrome_logits, treatment_logits, herb_logits
