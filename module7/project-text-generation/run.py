import torch

from config import *
from model import GeneratorModel
from tokenizer import CustomTokenizer
from utils import colorstr, generate, standardization

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load tokenizer
tokenizer = CustomTokenizer(VOCAB_SIZE, MAX_SEQ_LEN)
tokenizer.from_json(TOKENIZER_SAVE_PATH)

model = GeneratorModel(
    vocab_size=tokenizer.vocab_size,
    max_seq_len=MAX_SEQ_LEN,
    embedding_dim=EMBEDDING_DIM,
    n_heads=N_HEADS,
    ff_dim=FF_DIM,
    n_layers=N_LAYERS,
    dropout=DROP_OUT,
)
model.load_state_dict(PRETRAINED_PATH)
model = model.to(DEVICE)

# generate text
input_prompt = "I love"
output_prompt = generate(
    model, tokenizer, input_prompt, max_seq_len=MAX_SEQ_LEN, device=DEVICE
)

print(colorstr("blue", "bold", "Input prompt:"), input_prompt)
print(colorstr("blue", "bold", "Output prompt:"), output_prompt)