import torch
from torch.utils.data import Dataset

from dataset import TCMChainDataset
from symbolic_dataset import parse_token_field


class TCMNeuralSymbolicDataset(Dataset):
    def __init__(self, df, tokenizer, mappings, token_map, args):
        self.neural_dataset = TCMChainDataset(df, tokenizer, mappings, args)
        self.df = df.reset_index(drop=False).rename(columns={"index": "_sample_id"})
        self.token_map = token_map
        self.token_col = args.token_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.neural_dataset[idx]
        row = self.df.iloc[idx]

        token_indices = []
        for token_id in parse_token_field(row[self.token_col]):
            mapped = self.token_map.get(int(token_id))
            if mapped is not None:
                token_indices.append(mapped + 1)  # 0 is padding for symbolic token embedding.
        token_indices = sorted(set(token_indices))

        item["init_diag"] = item["western_diag"].clone()
        item["token_ids"] = torch.tensor(token_indices, dtype=torch.long)
        item["sample_id"] = torch.tensor(int(row["_sample_id"]), dtype=torch.long)
        return item


def neural_symbolic_collate_fn(batch):
    max_tokens = max((len(item["token_ids"]) for item in batch), default=0)
    max_tokens = max(max_tokens, 1)

    token_ids = torch.zeros(len(batch), max_tokens, dtype=torch.long)
    token_mask = torch.zeros(len(batch), max_tokens, dtype=torch.bool)

    for i, item in enumerate(batch):
        current = item["token_ids"]
        if len(current) == 0:
            continue
        token_ids[i, : len(current)] = current
        token_mask[i, : len(current)] = True

    fixed_keys = [
        "input_ids",
        "attention_mask",
        "age",
        "sex",
        "western_diag",
        "init_diag",
        "tcm_diag",
        "syndrome",
        "treatment",
        "herb",
        "sample_id",
    ]
    collated = {key: torch.stack([item[key] for item in batch]) for key in fixed_keys}
    collated["token_ids"] = token_ids
    collated["token_mask"] = token_mask
    return collated
