import os
import pickle
import numpy as np
import torch
from model import Transformer


# HYPERPARAMETERS
learning_rate = 1e-3
batch_size = 32
n_embd = 64
n_heads = 8
context_size = 16 
n_layer = 6
max_iter = 10000
eval_iter = 100
eval_interval = 500
dataset = 'homer'
device = 'cuda'

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
    data=train_idxs if split == "train" else eval_idxs
    ix = torch.randint(len(data) - context_size, (batch_size, ))
    x =  torch.stack([torch.from_numpy(data[i:i+context_size].astype(np.int64)) for i in ix])
    y =  torch.stack([torch.from_numpy(data[i+1:i+context_size+1].astype(np.int64)) for i in ix])
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
            B,T = x.shape
            assert B == batch_size and T==context_size, f"batch is {B} and context is {T} from eval"
            logits, loss = m(x,y)
            losses[i] = loss.item()
        out[split]=losses.mean()
    m.train()
    return out
            


# optimizer
m = Transformer(vocab_size=vocab_size, ctx_size=context_size, n_embd=n_embd, n_head=n_heads, n_layer=n_layer)
m.to(device)
adam = torch.optim.AdamW(m.parameters(), lr=learning_rate)
for i in range(max_iter):
    x,y=get_batch('train')
    B,T = x.shape
    assert B == batch_size and T==context_size, f"batch is {B} and context is {T} from training"
    logits, loss = m(x,y)
    if i%eval_interval==0:
        out = estimate_loss()
        print(f"epoch {i} training loss: {out['train']:.4f} - eval loss: {out['eval']:.4f}")
    loss.backward()
    adam.step()
    adam.zero_grad(set_to_none=True)



print("".join(itos[i.item()] for i in m.generate(torch.ones((1,1), dtype=torch.long, device=device))[0]))

