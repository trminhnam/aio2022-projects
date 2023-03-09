import matplotlib.pyplot as plt
import numpy as np
import torch
import base64
import os


def sliding_window(data, window_size, window_step):
    """Sliding window over a sequence of data.

    Args:
        data (list or dataframe): Data to be sliced.
        window_size (int): Size of the window.
        window_step (int): Step size of the window.
    """
    n = len(data)
    result = []
    for i in range(0, n - window_size + 1, window_step):
        result.append(np.array(data[i : i + window_size]))
    return result


def train_fn(model, dataloader, optimizer, criterion, device):
    model.train()
    model = model.to(device)
    losses = []
    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return sum(losses) / len(losses)


def eval_fn(model, dataloader, criterion, device):
    model.eval()
    model = model.to(device)
    losses = []
    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        loss = criterion(output, target)

        losses.append(loss.item())

    return sum(losses) / len(losses)


def plot_train_val_loss(train_losses, val_losses, title=None, filename=None):
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title or "Train and validation loss")
    if filename:
        plt.savefig(filename)
    plt.show()


def visualize_predictions(model, dataloader, device, target_idx, mean, std):
    model.eval()
    model = model.to(device)
    ground_truth = []
    predictions = []

    # plot the test true value and predicted value
    for batch_idx, (X_test, y_test) in enumerate(dataloader):
        X_test = X_test.to(device)
        y_test = y_test.to(device)

        with torch.no_grad():
            y_pred = model.predict(X_test)

        y_pred = y_pred.detach().cpu().numpy()
        y_test = y_test.detach().cpu().numpy()
        y_pred = y_pred * std[target_idx] + mean[target_idx]
        y_test = y_test * std[target_idx] + mean[target_idx]
        ground_truth.extend(y_test)
        predictions.extend(y_pred)

    plt.plot(ground_truth, label="true")
    plt.plot(predictions, label="pred")
    plt.legend()
    plt.show()
