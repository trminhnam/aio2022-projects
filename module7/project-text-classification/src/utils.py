import re
import string
import unicodedata

import torch
from tqdm.auto import tqdm


def unicodeToAscii(s):
    """Turn a Unicode string to plain ASCII
    Reference: https://stackoverflow.com/a/518232/2809427

    Args:
        s (str): input string to convert

    Returns:
        s: converted string
    """
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def normalizeString(text):
    """Normalize a string, including
    - removing URLs
    - removing punctuation
    - converting to lowercase
    - removing non-letter characters.

    Args:
        text (str): input string to normalize

    Returns:
        text: normalized string
    """
    text = re.sub(r"\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*", "", text)
    text = unicodeToAscii(text.lower().strip())
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", r" ", text).strip()
    return text


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
        mask = (
            batch.get("mask", None).to(device)
            if batch.get("mask", None) is not None
            else None
        )
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


def evaluate(model, test_loader, criterion, device, verbose=True):
    model.eval()
    model = model.to(device)
    epoch_loss = 0
    epoch_acc = 0
    step = 0
    pbar = (
        tqdm(test_loader, total=len(test_loader), disable=not verbose)
        if verbose
        else test_loader
    )
    with torch.no_grad():
        for batch in pbar:
            input = batch["input"].to(device)
            mask = (
                batch.get("mask", None).to(device)
                if batch.get("mask", None) is not None
                else None
            )
            target = batch["target"].to(device)

            output = model(input, mask)
            loss = criterion(output, target)
            acc = (output.argmax(1) == target).float().mean()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            step += 1

            if verbose:
                pbar.set_description(
                    f"Val loss: {epoch_loss / step:.4f} acc: {epoch_acc / step:.4f}"
                )

    return epoch_loss / len(test_loader), epoch_acc / len(test_loader)


def colorstr(*input):
    """Colorize a string with ANSI escape codes.
    Reference: https://github.com/ultralytics/yolov5/blob/4db6757ef9d43f49a780ff29deb06b28e96fbe84/utils/general.py#L684

    Returns:
        str: colored string
    """
    *args, string = (
        input if len(input) > 1 else ("blue", "bold", input[0])
    )  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]
