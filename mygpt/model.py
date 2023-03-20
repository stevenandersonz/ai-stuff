import math
import torch
import torch.nn as nn
from torch.nn import functional as F



class Head (nn.Module):
    def __init__(self,ctx_size, n_embd, head_size):
        super().__init__()
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('bias', torch.tril(torch.ones(ctx_size, ctx_size)))
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # B, T, T
        q = self.query(x)
        # k,q,v are of shape (batch_size, context_size, head_size)
        # q @ k won't work, k must me transpose so that it becomes (batch_size, head_size, context_size) 
        # q @ k computes the similarity between pairs of query and key vector

        # NOTE the dot product between two vectors measures the degree to which they point in the same direction. 
        # is positive and large, vectors are similar and pointing in the same direction. 
        # is zero, vectors are orthogonal and have no similarity. 
        # is negative, vectors point in opposite directions.
        wei = (q@k.transpose(-2,-1)) * (1/math.sqrt(k.shape[-1])) 
        # masked tokens affinity disabling communication between past and future tokens 
        wei = wei.masked_fill(self.bias[:T,:T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out

class MultiheadAttention (nn.Module):
    def __init__(self, n_head, ctx_size, n_embd, head_size):
        super().__init__()
        # n_heads must be 4x greater than head_size
        self.sa_heads = nn.ModuleList([Head(ctx_size, n_embd, head_size) for _ in range(n_head)])
        self.ffwd = FeedForwardNet(head_size*n_head)
        self.dropout = nn.Dropout(0.2)

    def forward(self,x):
        x = torch.cat([h(x) for h in self.sa_heads], dim=-1)
        x = self.dropout(self.ffwd(x))
        return x       

class FeedForwardNet (nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, n_embd*4),
            nn.Linear(n_embd*4, n_embd),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    def forward(self,x):
        return self.ffwd(x)

class Block(nn.Module):

    def __init__(self, n_embd, n_head, ctx_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiheadAttention(n_head=n_head, head_size=head_size, ctx_size=ctx_size, n_embd=n_embd)
        self.ffwd = FeedForwardNet(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
        

class Transformer (nn.Module):
    def __init__(self, vocab_size, ctx_size, n_embd, n_head, n_layer):
        super().__init__()
        self.ctx_size = ctx_size
        self.toks_embd = nn.Embedding(vocab_size, n_embd)
        self.pos_embd = nn.Embedding(ctx_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, ctx_size)for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)

    def forward(self, x, targets=None):
        # x should be of shape (batch_size, context_size)
        B,T=x.shape
        toks = self.toks_embd(x)
        pos =  self.pos_embd(torch.arange(T, device='cuda'))
        x = toks + pos
        x = self.blocks(x) 
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss=None
        if targets is not None:
            B,T,C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

        return logits,loss

    def generate(self, idx, max_token_len=100):
        for i in range(max_token_len):
            idx_cond = idx[:,-self.ctx_size:]
            logits, _ = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx,next_idx), dim=1)
        return idx
            
        
        