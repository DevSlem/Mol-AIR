import argparse
import os

import matplotlib.pyplot as plt
import torch
from selfies import get_alphabet_from_selfies
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from envs.selfies_tokenizer import SelfiesTokenizer
from train.net import SelfiesPretrainedNet
from util import TextInfoBox, seed


class SelfiesDataset(Dataset):
    def __init__(self, selfies_list):
        self.tokenizer = SelfiesTokenizer(vocabulary=get_alphabet_from_selfies(selfies_list))
        self.encoded_sequences = self.tokenizer.encode(selfies_list, include_stop_token=True)
        
    def __len__(self):
        return len(self.encoded_sequences)
    
    def __getitem__(self, idx):
        one_hot = self.tokenizer.to_one_hot(self.encoded_sequences[idx])
        one_hot = torch.from_numpy(one_hot).float()
        return one_hot[:-1], one_hot[1:]


def evaluate(net, dataloader, device):
    net.eval()
    losses = []
    H_shape = net.hidden_state_shape()
    with torch.no_grad():
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            mask = Y.sum(dim=-1) > 0
            dist, _ = net.forward(X, torch.zeros(H_shape[0], X.size(0), H_shape[1]).to(device))
            logits = dist._dist.logits[mask]
            target = Y[mask].argmax(dim=-1)
            loss = cross_entropy(logits, target)
            losses.append(loss.item())
    return sum(losses) / len(losses)

def save_pretrain(net, tokenizer, path):
    state_dict = {
        'model': net.state_dict(),
        'vocabulary': tokenizer.vocabulary,
    }
    torch.save(state_dict, path)

def train(args):
    if args.seed is not None:
        seed(args.seed)
    
    print(f"Building dataset from {args.data_path}...")
    with open(args.data_path) as f:
        selfies_list = f.read().splitlines()
    
    dataset = SelfiesDataset(selfies_list)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print("Building network...")
    device = torch.device(args.device)
    net = SelfiesPretrainedNet(vocab_size=dataset.tokenizer.vocab_size).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # Set up cosine annealing learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    text_info_box = TextInfoBox()
    text_info_box.add_text("SELFIES Pretraining Start!") \
        .add_line("=") \
        .add_text("Dataset:") \
        .add_text(f"    path: {args.data_path}") \
        .add_text(f"    # SELFIES: {len(dataset)}") \
        .add_text(f"    vocabulary size: {dataset.tokenizer.vocab_size}") \
        .add_line() \
        .add_text("Training INFO:") \
        .add_text(f"    epoch: {args.epoch}") \
        .add_text(f"    batch size: {args.batch_size}") \
        .add_text(f"    learning rate: {args.lr}") \
        .add_text(f"    device: {args.device}") \
        .add_text(f"    seed: {args.seed}") \
        .add_text(f"    output path: {args.output_path}") \
            
    print(text_info_box.make())
    os.makedirs(args.output_path, exist_ok=True)
        
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    lr_values = []
    H_shape = net.hidden_state_shape()
    
    for e in range(args.epoch):
        net.train()
        losses = []
        for X, Y in tqdm(train_dataloader, desc=f"Epoch {e+1}/{args.epoch}"):
            X, Y = X.to(device), Y.to(device)
            mask = Y.sum(dim=-1) > 0
            dist, _ = net.forward(X, torch.zeros(H_shape[0], X.size(0), H_shape[1]).to(device))
            logits = dist._dist.logits[mask]
            target = Y[mask].argmax(dim=-1)
            loss = cross_entropy(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        avg_train_loss = sum(losses) / len(losses)
        train_losses.append(avg_train_loss)
        avg_val_loss = evaluate(net, val_dataloader, device)
        val_losses.append(avg_val_loss)

        print(f"Epoch {e+1}/{args.epoch} - Train Loss: {avg_train_loss} | Val Loss: {avg_val_loss}")

        # Save checkpoint if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_pretrain(net, dataset.tokenizer, f"{args.output_path}/best.pt")

        # Update learning rate scheduler and record learning rate
        scheduler.step()
        lr_values.append(optimizer.param_groups[0]['lr'])
    
    # Save final model
    save_pretrain(net, dataset.tokenizer, f"{args.output_path}/final.pt")

    # Plot training and validation loss
    plt.title('Pretraining Loss')
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{args.output_path}/loss.png")
    plt.close()

    # Plot learning rate schedule
    plt.title('Learning Rate Schedule: Cosine Annealing')
    plt.plot(lr_values, label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.savefig(f"{args.output_path}/learning_rate.png")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_path', type=str, default="results/Pretraining")
    parser.add_argument('--seed', type=int, default=None)
    
    args = parser.parse_args()
    
    train(args)