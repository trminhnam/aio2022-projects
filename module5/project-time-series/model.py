import torch
import torch.nn as nn


class ForecastModel(nn.Module):
    def __init__(self, n_features, n_hidden, n_layers=2, batch_first=True):
        super(ForecastModel, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.batch_first = batch_first

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            dropout=0.5,
            batch_first=batch_first,
        )
        self.fc1 = nn.Linear(in_features=n_hidden, out_features=n_hidden // 2)
        self.batchnorm = nn.BatchNorm1d(n_hidden // 2)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(in_features=n_hidden // 2, out_features=1)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        if self.batch_first:
            lstm_out = lstm_out[:, -1, :]
        else:
            lstm_out = lstm_out[-1, :, :]
        lstm_out = lstm_out.view(-1, self.n_hidden).contiguous()
        out = self.fc1(lstm_out)
        out = self.batchnorm(out)
        out = self.activation(out)
        out = self.fc2(out)
        return out

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = self.forward(x)
        return x
