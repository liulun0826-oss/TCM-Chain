import torch
from torch.utils.data import Dataset
import pandas as pd
import ast

def parse_list_field(s):
    """Parse a serialized list field."""
    if isinstance(s, str) and s.startswith('[') and s.endswith(']'):
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return []
    return []

def parse_herb_dict_keys(s):
    """Parse a serialized dictionary field and return its keys."""
    if isinstance(s, str) and s.startswith('{') and s.endswith('}'):
        try:
            d = ast.literal_eval(s)
            return list(d.keys())
        except (ValueError, SyntaxError):
            return []
    return []

def _identity_map_from_observed(labels):
    labels = sorted({int(label) for label in labels if not pd.isna(label)})
    if not labels:
        return {}, {}
    max_label = max(labels)
    label_map = {label: label for label in range(max_label + 1)}
    inv_map = {label: label for label in range(max_label + 1)}
    return label_map, inv_map


def _identity_map_from_label_lists(label_lists):
    labels = set()
    for label_list in label_lists:
        labels.update(int(label) for label in label_list)
    if not labels:
        return {}, {}
    max_label = max(labels)
    label_map = {label: label for label in range(max_label + 1)}
    inv_map = {label: label for label in range(max_label + 1)}
    return label_map, inv_map


def build_label_mappings(df, tcm_diag_col, syndrome_col, treatment_col, herb_col, western_diag_col, mapping_mode="original"):
    """Build label-to-index mappings."""
    mappings = {}
    
    # TCM diagnosis
    all_tcm_diags = df[tcm_diag_col].dropna().unique()
    mappings['tcm_diag_map'] = {int(label): i for i, label in enumerate(all_tcm_diags)}
    mappings['tcm_diag_inv_map'] = {i: int(label) for i, label in enumerate(all_tcm_diags)}

    # Syndrome
    all_syndromes = df[syndrome_col].dropna().unique()
    mappings['syndrome_map'] = {int(label): i for i, label in enumerate(all_syndromes)}
    mappings['syndrome_inv_map'] = {i: int(label) for i, label in enumerate(all_syndromes)}

    # Initial diagnosis
    all_western_diags = df[western_diag_col].dropna().unique()
    mappings['western_diag_map'] = {int(label): i for i, label in enumerate(all_western_diags)}
    mappings['western_diag_inv_map'] = {i: int(label) for i, label in enumerate(all_western_diags)}

    # Treatment principles
    all_treatments = set()
    treatments = df[treatment_col].dropna().apply(parse_list_field)
    for treatment_list in treatments:
        all_treatments.update(treatment_list)
    mappings['treatment_map'] = {int(label): i for i, label in enumerate(sorted(list(all_treatments)))}
    mappings['treatment_inv_map'] = {i: int(label) for i, label in enumerate(sorted(list(all_treatments)))}

    # Herbs
    all_herbs = set()
    herbs = df[herb_col].dropna().apply(parse_herb_dict_keys)
    for herb_list in herbs:
        all_herbs.update(herb_list)
    mappings['herb_map'] = {int(label): i for i, label in enumerate(sorted(list(all_herbs)))}
    mappings['herb_inv_map'] = {i: int(label) for i, label in enumerate(sorted(list(all_herbs)))}

    if mapping_mode == "original":
        mappings['tcm_diag_map'], mappings['tcm_diag_inv_map'] = _identity_map_from_observed(df[tcm_diag_col].dropna().unique())
        mappings['syndrome_map'], mappings['syndrome_inv_map'] = _identity_map_from_observed(df[syndrome_col].dropna().unique())
        mappings['western_diag_map'], mappings['western_diag_inv_map'] = _identity_map_from_observed(df[western_diag_col].dropna().unique())
        mappings['treatment_map'], mappings['treatment_inv_map'] = _identity_map_from_label_lists(
            df[treatment_col].dropna().apply(parse_list_field)
        )
        mappings['herb_map'], mappings['herb_inv_map'] = _identity_map_from_label_lists(
            df[herb_col].dropna().apply(parse_herb_dict_keys)
        )
    elif mapping_mode != "compact":
        raise ValueError(f"Unsupported label mapping mode: {mapping_mode}")

    return mappings

class TCMChainDataset(Dataset):
    def __init__(self, df, tokenizer, mappings, args):
        self.df = df
        self.tokenizer = tokenizer
        self.mappings = mappings
        self.args = args

        self.text_col = args.text_col
        self.age_col = args.age_col
        self.sex_col = args.sex_col
        self.western_diag_col = args.western_diag_col
        self.tcm_diag_col = args.tcm_diag_col
        self.syndrome_col = args.syndrome_col
        self.treatment_col = args.treatment_col
        self.herb_col = args.herb_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Text
        text = row[self.text_col]
        text = "" if pd.isna(text) else str(text)
        
        inputs = self.tokenizer(text, max_length=self.args.max_length, padding='max_length', truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        # Structured features
        age_value = float(row[self.age_col])
        age_mean = getattr(self.args, 'age_mean', 0.0)
        age_std = getattr(self.args, 'age_std', 1.0)
        age = torch.tensor((age_value - age_mean) / age_std, dtype=torch.float32)
        sex = torch.tensor(row[self.sex_col], dtype=torch.long)
        western_diag = torch.tensor(self.mappings['western_diag_map'].get(int(row[self.western_diag_col]), -1), dtype=torch.long)

        # Labels
        tcm_diag = torch.tensor(self.mappings['tcm_diag_map'].get(int(row[self.tcm_diag_col]), -1), dtype=torch.long)
        syndrome = torch.tensor(self.mappings['syndrome_map'].get(int(row[self.syndrome_col]), -1), dtype=torch.long)

        # Multi-label treatment principles
        treatment_labels = parse_list_field(row[self.treatment_col])
        treatment_target = torch.zeros(len(self.mappings['treatment_map']))
        for label in treatment_labels:
            label = int(label)
            if label in self.mappings['treatment_map']:
                treatment_target[self.mappings['treatment_map'][label]] = 1.0

        # Multi-label herbs
        herb_labels = parse_herb_dict_keys(row[self.herb_col])
        herb_target = torch.zeros(len(self.mappings['herb_map']))
        for label in herb_labels:
            label = int(label)
            if label in self.mappings['herb_map']:
                herb_target[self.mappings['herb_map'][label]] = 1.0

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'age': age,
            'sex': sex,
            'western_diag': western_diag,
            'tcm_diag': tcm_diag,
            'syndrome': syndrome,
            'treatment': treatment_target,
            'herb': herb_target
        }
