import os
import pickle
import numpy as np



with open("./data.txt") as f:
    data = f.read()

vocab = "".join(sorted(list(set(data))))
vocab_size = len(vocab)
print(f"Unique characters: {vocab}")
print(f"vocab size: {vocab_size}")
stoi = {char:i for i, char in enumerate(vocab)}
itos = {i:char for char, i in stoi.items()}

def encode (s):
    return [stoi[c] for c in s]

def decode (idxs):
    return "".join([itos[idx] for idx in idxs])

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]
# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)




