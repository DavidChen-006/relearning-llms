"""train.py — pretrain the toy GLM on tiny-shakespeare.

Mirrors minimind/train_pretrain.py (structure, names, flow), minus the multi-GPU
armor (DDP/DistributedSampler) since this runs on one MacBook. Data comes from
data/train.bin + val.bin (prepare_data.py) instead of their jsonl-at-runtime path.
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset

from modeling_glm_moe_dsa import GlmMoeDsaConfig, GlmMoeDsaForCausalLM


def get_lr(current_step, total_steps, lr):   # minimind's schedule, verbatim (trainer_utils.py:40)
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def train_epoch(epoch, loader, iters):
    """Mirror of minimind's train_epoch: lr schedule -> forward -> grade -> backward
    -> clip -> step, with periodic logging. (No GradScaler: that's fp16/CUDA armor.)"""
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        logits = model(input_ids)                                # (B, S, vocab)
        loss = F.cross_entropy(logits.view(-1, lm_config.vocab_size), labels.view(-1))
        loss = loss / args.accumulation_steps

        loss.backward()

        if step % args.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            eta_min = spend_time / step * (iters - step) // 60
            print(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                  f"loss: {current_loss:.4f}, lr: {lr:.8f}, eta: {eta_min:.0f}min")

        if step >= args.max_steps:
            print(f"reached --max_steps {args.max_steps}, stopping")
            break


class PretrainDataset(Dataset):
    """Mirror of minimind's PretrainDataset, over a pre-tokenized .bin instead of jsonl.
    Sample i = window of ids starting at i; input/labels = the window shifted by one."""

    def __init__(self, data_path, max_length=256):
        self.ids = np.fromfile(data_path, dtype=np.uint16)
        self.max_length = max_length

    def __len__(self):
        return len(self.ids) - self.max_length - 1

    def __getitem__(self, index):
        window = self.ids[index : index + self.max_length + 1].astype(np.int64)
        input_ids = torch.from_numpy(window[:-1])   # tokens 0..255
        labels = torch.from_numpy(window[1:])       # tokens 1..256  (the "next token" answers)
        return input_ids, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Toy GLM Pretraining")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str,
                        default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--num_hidden_layers", default=2, type=int)
    parser.add_argument("--max_seq_len", default=256, type=int)
    parser.add_argument("--data_path", type=str, default="data/train.bin")
    parser.add_argument("--max_steps", type=int, default=10**9, help="stop early (smoke tests)")
    args = parser.parse_args()

    # ========== 1. seed ==========
    torch.manual_seed(42)

    # ========== 2. dirs + model config ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = GlmMoeDsaConfig(
        vocab_size=6400,                       # must match tokenizer.json
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=4,
        intermediate_size=args.hidden_size * 2,
    )
    lm_config._attn_implementation = "eager"

    # ========== 5. model, data, optimizer ==========
    model = GlmMoeDsaForCausalLM(lm_config).to(args.device)
    model.train()
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}  device: {args.device}")

    train_ds = PretrainDataset(args.data_path, max_length=args.max_seq_len)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 8. train ==========
    iters = min(len(loader), args.max_steps)
    for epoch in range(args.epochs):
        train_epoch(epoch, loader, iters)
