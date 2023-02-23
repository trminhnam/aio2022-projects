import os
import random
import re
import string

import numpy as np
import torch
import torch.nn as nn
from tqdm.notebook import tqdm

from dataset import TextGenerationDataset
from model import GeneratorModel
from tokenizer import CustomTokenizer
from train import train
from utils import generate, standardization, colorstr
from config import *

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
CLASSES = ["neg", "pos"]

OUTPUT_DIR = "output"
MODEL_PATH = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists(OUTPUT_DIR) is False:
    os.mkdir(OUTPUT_DIR)

file_path = []
for SUBSET_DIR in [TRAIN_DIR, TEST_DIR]:
    for CLASS in CLASSES:
        CLASS_DIR = os.path.join(SUBSET_DIR, CLASS)
        for filename in os.listdir(CLASS_DIR):
            file_path.append(os.path.join(CLASS_DIR, filename))
print(colorstr("blue", "bold", "Number of text files:"), len(file_path))

tokenizer = CustomTokenizer(VOCAB_SIZE, MAX_SEQ_LEN)
tokenizer.fit_corpus(
    [standardization(open(path, "r", encoding="utf8").read()) for path in file_path]
)
print(colorstr("blue", "bold", "Vocab size:"), tokenizer.vocab_size)

print(colorstr("yellow", "bold", "Tokenizer example:"))
text = "this is a test sentence"
encoded_text, mask = tokenizer.encode(text, add_sos=True, get_mask=True)
print(colorstr("blue", "bold", "Encoded text:"), f"{encoded_text}")
print(colorstr("blue", "bold", "Mask:"), f"{mask}")
print(colorstr("blue", "bold", "Decoded text:"), tokenizer.decode(encoded_text))
print()


dataset = TextGenerationDataset(file_path, MAX_SEQ_LEN, tokenizer)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(colorstr("yellow", "bold", "Training dataset example:"))
idx = random.randint(0, len(dataset))
sample = dataset[idx]
encoded_input = sample["input_ids"]
encoded_target = sample["target_ids"]
print(colorstr("blue", "bold", "Encoded input: "))
print(encoded_input)
print(colorstr("blue", "bold", "Encoded target: "))
print(encoded_target)
print(colorstr("blue", "bold", "Decoded input: "))
print(tokenizer.decode(encoded_input))
print()

model = GeneratorModel(
    vocab_size=tokenizer.vocab_size,
    max_seq_len=MAX_SEQ_LEN,
    embedding_dim=EMBEDDING_DIM,
    n_heads=N_HEADS,
    ff_dim=FF_DIM,
    n_layers=N_LAYERS,
    dropout=DROP_OUT,
)
if MODEL_PATH is not None:
    model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
critetion = nn.CrossEntropyLoss(ignore_index=tokenizer["<PAD>"])

print(colorstr("yellow", "bold", "Training..."))
for epoch in range(EPOCHS):
    print(colorstr("cyan", "bold", "Training Epoch:"), f"{epoch+1:02}")
    train_loss = train(model, dataloader, optimizer, critetion, DEVICE)
    print(f"Epoch: {epoch+1:02}, Train Loss: {train_loss:.4f}")
    print("#" * 100)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"model_{epoch+1:02}.pt"))

start_prompt = "this movie is"
start_tokens = tokenizer.encode(start_prompt)[: len(start_prompt.split())]
num_tokens_generated = 30
self_max_tokens = 30

# generated_text = generate(start_tokens, max_generated_tokens=self_max_tokens)
generated_text = generate(
    model,
    tokenizer,
    start_tokens,
    max_generated_tokens=self_max_tokens,
    max_seq_len=MAX_SEQ_LEN,
)

print(colorstr("yellow", "bold", "Start prompt:"), start_prompt)
print(colorstr("yellow", "bold", "Generated text:"), generated_text)
