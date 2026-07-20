import argparse
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

    # tokenizer.json lives in ../ (david/), anchored to this file so launch dir doesn't matter
    tokenizer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "pretrain_shakespeare.jsonl")
    train_ds = PretrainDataset(data_path, tokenizer, max_length=args.max_seq_len)

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
    )
    lm_config._attn_implementation = "eager"


    model = GlmMoeDsaForCausalLM(lm_config)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

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
