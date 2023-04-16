"""This file will import the data, clean it, tokenize it token-wise and train it"""
import sys
sys.path.append('/notebooks/') # path to main directory of this project
print(sys.path)
import ftfy
import re
from nltk.tokenize import word_tokenize
import nltk
import torch
from tqdm import tqdm
import time
from model import DecoderTransformer
# import os
# print(os.path())

text = open("/notebooks/data_files/dickens_in_one_place.txt", "r").read()
print("length of the dataset:", len(text))



text = ftfy.fix_text(text)
text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", text)


nltk.download('punkt')
text_words = word_tokenize(text.lower())
text_words[:10]

# chars = sorted(list(set(text)))
# vocab_size = len(chars)
# print(''.join(chars))
# print(vocab_size)


# encoding and decoding the data
import tiktoken
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

# print(encode(["hi", "there"]))
# print(decode(encode(["hi", "there"])))
encode = lambda s:enc.encode(s)
decode = lambda l:" ".join(enc.decode([i]) for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
NUM_EPOCHS = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 8
dropout = 0.25
# ------------

# data loading
block_size = 256
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


model = DecoderTransformer(vocab_size).to(device)


learning_rate = 3e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
all_losses = []

for e in range(NUM_EPOCHS):
    if e % eval_interval == 0:
        losses = estimate_loss()
        print(f'step {e}: train loss {losses["train"]}, val loss {losses["val"]}')

    # set the model in training mode
    model.train()

    # initialize the total training and validation loss
    totalTrainLoss = 0
    totalTestLoss = 0
    # loop over the training set
    xb, yb = get_batch("train")

    logits, loss = model(xb.to(device), yb.to(device))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    all_losses.append(loss.item())


# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))

torch.save(model, "trained_models/word_level_model_trained.pb")

print("/n/nFirst 500 tokens generated by the model:")
print(decode(model.generate(idx=torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=500)[0].tolist()))