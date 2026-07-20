import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

from torch import optim

# modeling_glm_moe_dsa lives in ../architecture, not next to this file
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "architecture"))

from transformers import AutoTokenizer

from lm_dataset import PretrainDataset
from modeling_glm_moe_dsa import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

def train_epoch(epoch, loader, iters): #training loop

    for step, (input_ids, labels) in enumerate(loader, start=1):

        res = model(input_ids, labels=labels)   #forward
        loss = res.loss

        #backprop
        loss.backward()

        #gradient descent, update weights: w  ←  w  −  lr · (∂L/∂w) + AdamW optimizer
        optimizer.step()

        #clean
        optimizer.zero_grad()

        if step % 10 == 0:
            print(f"epoch {epoch}  step {step}/{iters}  loss {loss.item():.4f}")
            # one dict of scalars per logging step — same shape wandb.log() takes later
            with open(log_path, "a") as f:
                f.write(json.dumps({"epoch": epoch, "step": step, "train_loss": loss.item()}) + "\n")

        if step % 100 == 0:
            val = estimate_val_loss()
            print(f"epoch {epoch}  step {step}/{iters}  VAL loss {val:.4f}")
            with open(log_path, "a") as f:
                f.write(json.dumps({"epoch": epoch, "step": step, "val_loss": val}) + "\n")
            model.train()


@torch.no_grad()
def estimate_val_loss(num_batches=10):
    """Grade the model on UNSEEN text (val_shakespeare.jsonl) — measured, never learned from."""
    model.eval()
    losses = []
    for i, (input_ids, labels) in enumerate(val_loader):
        if i >= num_batches:
            break
        losses.append(model(input_ids, labels=labels).loss.item())
    return sum(losses) / len(losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="David GLM Pretraining")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--num_hidden_layers", default=2, type=int)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument("--save_dir", default="out", type=str)        # minimind: --save_dir
    parser.add_argument("--save_weight", default="pretrain", type=str)  # minimind: --save_weight
    parser.add_argument("--mlp", default="dense", choices=["dense", "moe"],
                        help="ablation switch: dense = all-dense layers; moe = layer 0 dense, rest sparse")

    args = parser.parse_args()

    torch.manual_seed(42)   # same seed both runs — identical init + batch order, only --mlp differs

    # tokenizer.json lives in ../ (david/), anchored to this file so launch dir doesn't matter
    tokenizer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "pretrain_shakespeare.jsonl")
    train_ds = PretrainDataset(data_path, tokenizer, max_length=args.max_seq_len)

    val_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "val_shakespeare.jsonl")
    val_ds = PretrainDataset(val_path, tokenizer, max_length=args.max_seq_len)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)   # fixed order: comparable val numbers

    # the experiment variable: identical code both runs, only this differs
    if args.mlp == "moe":
        mlp_layer_types = ["dense"] + ["sparse"] * (args.num_hidden_layers - 1)
    else:
        mlp_layer_types = ["dense"] * args.num_hidden_layers

    lm_config = GlmMoeDsaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=4,
        intermediate_size=args.hidden_size * 2,
        mlp_layer_types=mlp_layer_types,
        # matched ACTIVE compute vs dense (256): top_k 2 x 64 + shared 2 x 64 = 256 per token.
        # total stored capacity is bigger (8 experts x 64 = 512) — that's the MoE claim under test.
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        n_shared_experts=2,
    )
    lm_config._attn_implementation = "eager"


    model = GlmMoeDsaForCausalLM(lm_config)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # metrics file named from the experiment flag: runs/dense.jsonl or runs/moe.jsonl
    runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    log_path = os.path.join(runs_dir, f"{args.mlp}.jsonl")
    open(log_path, "w").close()   # fresh file each run — a run's log is one run's data

    for epoch in range(args.epochs):   #outer training loop
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)   # fresh shuffled batches
        train_epoch(epoch, loader, len(loader))                             # run the inner loop

    # save the trained weights (mirrors minimind train_pretrain.py:62-68, minus the
    # DDP/scaler/half armor — half() is a GPU trick that doesn't run on CPU)
    model.eval()
    os.makedirs(args.save_dir, exist_ok=True)
    ckp = f"{args.save_dir}/{args.save_weight}_{args.hidden_size}.pth"   # minimind's filename convention
    torch.save(model.state_dict(), ckp)
    print(f"saved weights -> {ckp}")
