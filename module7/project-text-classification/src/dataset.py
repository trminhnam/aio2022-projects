import torch
from torch.utils.data import Dataset


class TextClassificationDataset(Dataset):
    def __init__(self, data, targets, tokenizer, max_len):
        self.data = data
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        target = self.targets[idx]

        encoded_text, mask = self.tokenizer.encode(text, get_mask=True)

        if not isinstance(encoded_text, torch.Tensor):
            encoded_text = torch.tensor(encoded_text, dtype=torch.long)
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.long)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.long)

        return {"input": encoded_text, "mask": mask, "target": target}
