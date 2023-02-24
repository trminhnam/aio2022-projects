import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import config as cfg
from src.dataset import TextClassificationDataset
from src.model import TextClassificationModel
from src.tokenizer import WordTokenizer
from src.utils import evaluate, normalizeString, train

train_df = pd.read_csv("./dataset/disaster-twitter/train.csv")
corpus = [normalizeString(text) for text in train_df.text.values]
target = train_df.target.values

X_train, X_val, y_train, y_val = train_test_split(
    corpus, target, test_size=0.2, random_state=42
)


print("Some examples of training data:")
idx = random.randint(0, len(X_train))
print(f"X_train[{idx}]:", X_train[idx])
print(f"y_train[{idx}]:", y_train[idx])

tokenizer = WordTokenizer(cfg.VOCAB_SIZE, cfg.MAX_SEQ_LENGTH)

tokenizer.add_corpus(corpus)


example = "I am a student"
example = normalizeString(example)

print("Example:", example)
ids, mask = tokenizer.encode(example, get_mask=True)
print("Encoded:", ids)
print("Decoded:", tokenizer.decode(ids))


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


train_dataset = TextClassificationDataset(
    X_train, y_train, tokenizer, cfg.MAX_SEQ_LENGTH
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True
)

test_dataset = TextClassificationDataset(X_val, y_val, tokenizer, cfg.MAX_SEQ_LENGTH)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False
)

optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(cfg.NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{cfg.NUM_EPOCHS}")
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
    evaluate_loss, evaluate_acc = evaluate(
        model, test_loader, criterion, device, verbose=True
    )


torch.save(model.state_dict(), "./model/model.pt")
