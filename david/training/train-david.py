import argparse

from torch.utils.data import DataLoader

from lm_dataset import PretrainDataset

def train_epoch(epoch, loader, iters): #training loop

    for step, (input_ids, labels) in enumerate(loader, start=1):
        
        res = model(input_ids, labels=labels)   #forward
        loss = res.loss

        #gradient 
        loss.backward()

        #nudge

        #clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="David GLM Pretraining")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")

    train_ds = PretrainDataset()


    for epoch in range(args.epochs):   #outer training loop
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)   # fresh shuffled batches
        train_epoch(epoch, loader, len(loader))                             # run the inner loop
