"""lm_dataset.py — dataset classes for training (mirrors minimind/lm_dataset.py).

Only PretrainDataset for now; the other classes (SFTDataset, DPODataset, ...) arrive
when their training phases do.
"""
import os

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        tokens = self.tokenizer(
            str(sample['text']), 
            add_special_tokens=False, 
            max_length=self.max_length - 2, 
            truncation=True).input_ids

        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return input_ids, labels


if __name__ == "__main__":   # self-check, mirroring minimind's lm_dataset.py
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("..")
    ds = PretrainDataset("data/pretrain_shakespeare.jsonl", tok, max_length=512)

    input_ids, labels = ds[0]
    
    n_real = (input_ids != tok.pad_token_id).sum().item()
    assert input_ids[0].item() == tok.bos_token_id, "must start with BOS"
    assert input_ids[n_real - 1].item() == tok.eos_token_id, "EOS must sit right before padding"
    assert ((labels == -100) == (input_ids == tok.pad_token_id)).all(), "-100 exactly on padding"
    print(f"documents: {len(ds)}  sample0: {n_real} real tokens + {len(input_ids) - n_real} pad")
    print("decoded:", repr(tok.decode(input_ids[:20])))
    print("PASS")
