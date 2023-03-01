# %%
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import config as cfg
from dataset import TextClassificationDataset
from tokenizer import WordTokenizer
from utils import normalizeString

# %% [markdown]
# ## Load dataset

# %% [markdown]
# ### Disaster Tweets

# %%
train_df = pd.read_csv("./dataset/disaster-twitter/train.csv")
# test_df = pd.read_csv("./dataset/disaster-twitter/test.csv")

train_df.head()

# %%
corpus = [normalizeString(text) for text in train_df.text.values]
target = train_df.target.values

X_train, X_val, y_train, y_val = train_test_split(
    corpus, target, test_size=0.2, random_state=42
)

# %%
print("Some examples of training data:")
idx = random.randint(0, len(X_train))
print(f"X_train[{idx}]:", X_train[idx])
print(f"y_train[{idx}]:", y_train[idx])

# %% [markdown]
# ### ABCDEF

# %% [markdown]
# ## Tokenizer

# %%
tokenizer = WordTokenizer(cfg.VOCAB_SIZE, cfg.MAX_SEQ_LENGTH)

tokenizer.add_corpus(corpus)

# %%
example = "I am a student"
example = normalizeString(example)

print("Example:", example)
ids, mask = tokenizer.encode(example, get_mask=True)
print("Encoded:", ids)
print("Decoded:", tokenizer.decode(ids))

# %% [markdown]
# ## Model

# %%
from model import TextClassificationModel

model = TextClassificationModel(
    vocab_size=len(tokenizer.word2index),
    num_classes=2,
    max_seq_len=cfg.MAX_SEQ_LENGTH,
    embedding_dim=cfg.EMBEDDING_DIM,
    n_heads=cfg.N_HEADS,
    n_layers=cfg.N_LAYERS,
    ff_dim=cfg.FF_DIM,
    drop_out=cfg.DROP_OUT,
)

# %% [markdown]
# ## Training


# %%
def train(model, train_loader, optimizer, criterion, device, verbose=True):
    model.train()
    model = model.to(device)
    epoch_loss = 0
    epoch_acc = 0
    step = 0
    pbar = (
        tqdm(train_loader, total=len(train_loader), disable=not verbose)
        if verbose
        else train_loader
    )
    for batch in pbar:
        optimizer.zero_grad()
        input = batch["input"].to(device)
        mask = batch["mask"].to(device)
        target = batch["target"].to(device)

        output = model(input, mask)
        loss = criterion(output, target)
        acc = (output.argmax(1) == target).float().mean()

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        step += 1

        if verbose:
            pbar.set_description(
                f"Train loss: {epoch_loss / step:.4f} acc: {epoch_acc / step:.4f}"
            )

    return epoch_loss / len(train_loader), epoch_acc / len(train_loader)


# %%
train_dataset = TextClassificationDataset(
    X_train, y_train, tokenizer, cfg.MAX_SEQ_LENGTH
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True
)

optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(cfg.NUM_EPOCHS):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)

# %%
