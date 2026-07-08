import argparse

import torch
from torch.utils.data import DataLoader

from torch import optim

from lm_dataset import PretrainDataset

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="David GLM Pretraining")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")

    train_ds = PretrainDataset()

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):   #outer training loop
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)   # fresh shuffled batches
        train_epoch(epoch, loader, len(loader))                             # run the inner loop
