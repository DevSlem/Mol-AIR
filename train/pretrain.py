from typing import List, Optional

import selfies as sf
import torch
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from drl.agent import PretrainedRecurrentAgent, PretrainedRecurrentNetwork
from envs.chem_env import make_async_chem_env
from envs.selfies_tokenizer import SelfiesTokenizer
from train.train import Train
from util import (load_smiles_or_selfies, load_vocab, logger, save_vocab,
                  to_selfies, try_create_dir)


class SelfiesDataset(Dataset):
    def __init__(self, selfies_list: List[str]):
        self.tokenizer = SelfiesTokenizer(vocabulary=sf.get_alphabet_from_selfies(selfies_list))
        self.encoded_sequences = self.tokenizer.encode(selfies_list, include_stop_token=True)
        
    def __len__(self):
        return len(self.encoded_sequences)
    
    def __getitem__(self, idx):
        one_hot = self.tokenizer.to_one_hot(self.encoded_sequences[idx])
        one_hot = torch.from_numpy(one_hot).float()
        return one_hot[:-1], one_hot[1:]
    
    @staticmethod
    def from_txt(file_path: str, auto_convert: bool = True) -> "SelfiesDataset":
        smiles_or_selfies_list = load_smiles_or_selfies(file_path)
        selfies_list = to_selfies(smiles_or_selfies_list) if auto_convert else smiles_or_selfies_list
        return SelfiesDataset(selfies_list)

class Pretrain:
    def __init__(
        self,
        id: str,
        net: PretrainedRecurrentNetwork,
        dataset: SelfiesDataset,
        epoch: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: Optional[str] = None,
        save_agent: bool = True
    ) -> None:
        self._id = id
        self._net = net
        self._dataset = dataset
        self._epoch = epoch
        self._batch_size = batch_size
        self._lr = lr
        self._device = torch.device(device) if device is not None else torch.device("cpu")
        self._save_agent = save_agent
        
        self._enabled = True
        
    def pretrain(self) -> "Pretrain":
        if not self._enabled:
            raise RuntimeError("Pretrain is already closed.")       
        
        logger.enable(self._id, enable_log_file=True)

        train_size = int(0.9 * len(self._dataset))
        val_size = len(self._dataset) - train_size
        train_dataset, val_dataset = random_split(self._dataset, [train_size, val_size])
        
        train_dataloader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self._batch_size, shuffle=False)
        save_vocab(self._dataset.tokenizer.vocabulary, self._dataset.encoded_sequences.shape[1] - 1, f"{logger.dir()}/vocab.json")
        
        self._net.model().to(self._device)
        optimizer = torch.optim.Adam(self._net.model().parameters(), lr=self._lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self._epoch)
        
        logger.print("SELFIES Pretraining start!")
        
        best_val_loss = float('inf')
        H_shape = self._net.hidden_state_shape()
        
        for e in range(self._epoch):
            self._net.model().train()
            losses = []
            for X, Y in tqdm(train_dataloader, desc=f"Epoch {e+1}/{self._epoch}"):
                X, Y = X.to(self._device), Y.to(self._device)
                mask = Y.sum(dim=-1) > 0
                dist, _ = self._net.forward(X, torch.zeros(H_shape[0], X.size(0), H_shape[1]).to(self._device))
                logits = dist._dist.logits[mask] # type: ignore
                target = Y[mask].argmax(dim=-1)
                loss = cross_entropy(logits, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            
            avg_train_loss = sum(losses) / len(losses)
            avg_val_loss = self._evaluate(val_dataloader)

            logger.print(f"Epoch {e+1}/{self._epoch} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            logger.log_data("Pretrain/Train Loss", avg_train_loss, e)
            logger.log_data("Pretrain/Val Loss", avg_val_loss, e)

            # Save checkpoint if validation loss improves
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self._save_pretrain("best.pt")

            # Update learning rate scheduler and record learning rate
            scheduler.step()
            logger.log_data("Pretrain/Learning Rate", optimizer.param_groups[0]['lr'], e)
            
            logger.plot_logs()
            
        self._save_pretrain("final.pt")
        self._save_pretrained_agent()
        
        return self
        
    def close(self) -> None:
        self._enabled = False
        
        if logger.enabled():
            logger.disable()
            
    def _evaluate(self, dataloader):
        self._net.model().eval()
        losses = []
        H_shape = self._net.hidden_state_shape()
        with torch.no_grad():
            for X, Y in tqdm(dataloader, desc="Validation"):
                X, Y = X.to(self._device), Y.to(self._device)
                mask = Y.sum(dim=-1) > 0
                dist, _ = self._net.forward(X, torch.zeros(H_shape[0], X.size(0), H_shape[1]).to(self._device))
                logits = dist._dist.logits[mask] # type: ignore
                target = Y[mask].argmax(dim=-1)
                loss = cross_entropy(logits, target)
                losses.append(loss.item())
        return sum(losses) / len(losses)
    
    def _save_pretrain(self, file_name):
        try_create_dir(f"{logger.dir()}/pretrained_models")
        state_dict = {
            'model': self._net.model().state_dict(),
            # 'vocabulary': self._dataset.tokenizer.vocabulary,
        }
        torch.save(state_dict, f"{logger.dir()}/pretrained_models/{file_name}")
        
    def _save_pretrained_agent(self):
        if not self._save_agent:
            return
        
        state_dict = torch.load(f"{logger.dir()}/pretrained_models/best.pt")
        vocab, max_str_len = load_vocab(f"{logger.dir()}/vocab.json")
        env = make_async_chem_env(
            num_envs=1,
            seed=None,
            vocabulary=vocab,
            max_str_len=max_str_len
        )
        self._net.model().load_state_dict(state_dict['model'])
        agent = PretrainedRecurrentAgent(self._net, num_envs=1)
        # just save the agent
        Train(env, agent, self._id, total_time_steps=0).train().close()