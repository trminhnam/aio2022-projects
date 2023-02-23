import numpy as np
from tqdm.auto import tqdm


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
