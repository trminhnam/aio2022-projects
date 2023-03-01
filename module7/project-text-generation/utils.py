import re
import string

import numpy as np
import torch
from tqdm.auto import tqdm


def remove_html_tags(text):
    """Remove html tags from a string
    Reference: https://stackoverflow.com/a/9662362

    Args:
        text (str): input string/document

    Returns:
        str: string without html tags

    """
    TAG_RE = re.compile(r"<[^>]+>")
    return TAG_RE.sub("", text)


def standardization(text):
    # lowercase
    text = text.lower()

    # remove newline
    text = text.replace("\n", " ").replace("\r", "")

    # remove html tags
    text = remove_html_tags(text)

    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # remove extra spaces
    text = re.sub(" +", " ", text)

    return text


def prepare_lm_input_labels(text, tokenizer):
    input_ids, input_mask = tokenizer.encode(text, add_sos=True, get_mask=True)
    labels, label_mask = tokenizer.encode(text, add_sos=False, get_mask=True)
    return input_ids, input_mask, labels, label_mask


def get_casual_attention_mask(batch_size, seq_len, n_dest, n_src, dtype=torch.float32):
    """Generate a causal mask for the decoder

    Args:
        batch_size (int): batch size
        seq_len (int): sequence length
        n_dest (int): number of destination tokens
        n_src (int): number of source tokens

    Returns:
        torch.Tensor: causal mask
    """
    i = torch.arange(n_dest)[:, None]
    j = torch.arange(n_src)
    m = i >= j - n_src + n_dest
    mask = m.to(dtype)
    mask = mask.reshape(1, n_dest, n_src)
    mult = torch.tensor([batch_size, 1, 1])
    return mask.repeat(mult.tolist())


def sample_from(logits, top_k=10):
    logits, indices = torch.topk(logits, top_k)
    indices = np.asarray(indices).astype(np.int32)
    preds = torch.softmax(logits, dim=-1).numpy()
    preds = np.asarray(preds).astype(np.float32)
    return np.random.choice(indices, p=preds)


def generate(start_prompt, model, tokenizer, max_generated_tokens, max_seq_len):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    start_tokens = tokenizer.encode(start_prompt)[:len(start_prompt.split())]
    start_tokens = [_ for _ in start_tokens]
    num_tokens_generated_local = 0
    tokens_generated = []
    max_generated_tokens 
    while num_tokens_generated_local <= max_generated_tokens:
        pad_len = max_seq_len - len(start_tokens)
        sample_index = len(start_tokens) - 1
        if pad_len > 0:
            x = start_tokens + [0] * pad_len
        else:
            x = start_tokens
        x = torch.tensor(x).unsqueeze(0).to(device)
        with torch.no_grad():
            y = model(x)
            y = y.cpu()
            
        sample_token = sample_from(y[0][sample_index])
        
        if sample_token == tokenizer['<EOS>']:
            break
        
        tokens_generated.append(sample_token)
        start_tokens.append(sample_token)
        num_tokens_generated_local = len(tokens_generated)
        
    txt = tokenizer.decode(tokens_generated)
    return txt

def train(model, data_loader, optimizer, criterion, device):
    model.train()
    losses = []
    avg_loss = 0
    pbar = tqdm(data_loader)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        input_mask = batch["input_mask"].to(device)
        target_mask = batch["target_mask"].to(device)

        optimizer.zero_grad()
        output = model(input_ids, input_mask)
        output = output.reshape(-1, output.shape[-1])
        target_ids = target_ids.reshape(-1)
        target_mask = target_mask.reshape(-1)

        loss = criterion(output, target_ids)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        avg_loss = (0.99 * avg_loss + 0.01 * loss.item()) / (
            1 - 0.99 ** (len(losses) + 1)
        )
        pbar.set_description(f"Loss: {loss.item():.4f}")
    return np.mean(losses)

def colorstr(*input):
    """Colorize a string with ANSI escape codes.
    Reference: https://github.com/ultralytics/yolov5/blob/4db6757ef9d43f49a780ff29deb06b28e96fbe84/utils/general.py#L684

    Returns:
        str: colored string
    """
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag.
    Reference: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Args:
        sz (int): the size of the mask

    Returns:
        torch.Tensor: the mask tensor
    """
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)