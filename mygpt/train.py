import os
import pickle
import numpy as np
import torch
from model import Transformer, TransformerConfig


# HYPERPARAMETERS
learning_rate = 1e-3
batch_size = 64
max_iter = 15000
eval_iter = 100
eval_interval = 500
dataset = 'homer'
device = 'cuda'
config = TransformerConfig()
load_path = os.path.join('data',dataset)
#Loading dataset
train_idxs = np.memmap(os.path.join(load_path, 'train.bin'), dtype=np.uint16, mode='r')
eval_idxs = np.memmap(os.path.join(load_path, 'train.bin'), dtype=np.uint16, mode='r')

#Loading metadata from dataset
with open(os.path.join(load_path, 'meta.pkl'), 'rb') as f:
    ds_meta = pickle.load(f)

stoi, itos, vocab_size = ds_meta["stoi"], ds_meta["itos"], ds_meta["vocab_size"] 

#Prepare dataset
def get_batch(split):
    block_size = config.block_size
    data=train_idxs if split == "train" else eval_idxs
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x =  torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y =  torch.stack([torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)) for i in ix])
    x,y = x.to(device),y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    m.eval()
    out = {}
    for split in ['train', 'eval']:
        losses = torch.zeros(eval_iter)
        for i in range(eval_iter):
            x, y = get_batch(split)
            logits, loss = m(x,y)
            losses[i] = loss.item()
        out[split]=losses.mean()
    m.train()
    return out
            


# optimizer
m = Transformer(config)
m.to(device)
adam = torch.optim.AdamW(m.parameters(), lr=learning_rate)
import time
start_time = time.time()
for i in range(max_iter):
    x,y=get_batch('train')
    B,T = x.shape
    logits, loss = m(x,y)
    if i%eval_interval==0:
        out = estimate_loss()
        print(f"epoch {i} training loss: {out['train']:.4f} - eval loss: {out['eval']:.4f}")
    loss.backward()
    adam.step()
    adam.zero_grad(set_to_none=True)
end_time = time.time()
print(f"Execution time: {end_time - start_time:.2f} seconds")
print(f"Each loop on avg time: {(end_time - start_time)/max_iter:.2f} seconds")
print("".join(itos[i.item()] for i in m.generate(torch.ones((1,1), dtype=torch.long, device=device))[0]))

