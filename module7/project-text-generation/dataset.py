import torch

from utils import prepare_lm_input_labels, standardization


class TextGenerationDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, seq_len, tokenizer, standardize=True, get_mask=True):
        self.file_paths = file_paths
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.standardize = standardize
        self.get_mask = get_mask

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with open(self.file_paths[idx], "r", encoding="utf-8") as f:
            text = f.read()

        if self.standardize:
            text = standardization(text)

        input_ids, input_mask, labels, label_mask = prepare_lm_input_labels(
            text, self.tokenizer
        )
        sample = {
            "input_ids": input_ids,
            "target_ids": labels,
        }
        if self.get_mask:
            sample["input_mask"] = input_mask
            sample["target_mask"] = label_mask

        return sample

        # return self.tokenizer.encode(text, add_sos=True, get_mask=self.get_mask)
