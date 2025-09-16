import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

"""
Input Formats:
    - X: (N, K, 1) 
    where X[i, j, 0] is the jth bin retweets number of the ith cascade
    - Y: N tuples of (lambda, K) of weibull distribution parameters
    where Y[i][0] is the lambda of the ith cascade and Y[i][1] is the K of the ith cascade
Model Summary:
    we use a simple LSTM model to predict the lambda and K of the weibull distribution
    the input is the retweets number of each bin
    the output is the lambda and K of the weibull distribution 
"""


class RecurrentModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RecurrentModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.GRU(input_size, hidden_size, batch_first=True)
        self.hidden2lambda = nn.Linear(hidden_size, 1)
        self.hidden2K = nn.Linear(hidden_size, 1)
        self.softPlus_layer = nn.Softplus()

    def forward(self, input, hidden):
        input = input.unsqueeze(-1)
        lstm_out, hidden = self.lstm(input, hidden)
        fc_input = lstm_out[:, -1, :]
        lambda_out = self.hidden2lambda(fc_input, )
        K_out = self.hidden2K(fc_input, )
        K_out = self.softPlus_layer(K_out)
        lambda_out = torch.exp(lambda_out)
        return lambda_out, K_out

    def init_hidden(self, batch_size, device):
        #return (torch.zeros(1, batch_size, self.hidden_size, device=device),
        #         torch.zeros(1, batch_size, self.hidden_size, device=device))
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


def loss_function(lambdas, k, labels, times) -> torch.Tensor:
    """
    this is run on all time steps of the cascade N (N > K)
    """
    log_survival = -torch.pow(times / lambdas, k)  # TODO: Broadcasting Checked? Checked:TRUE
    inner_cost = torch.log(k / lambdas) + (k - 1) * torch.log(times / lambdas)
    final_matrix = log_survival + inner_cost * labels
    loss = -torch.sum(final_matrix)
    return loss


def loss_function_exponential(lambdas, k, labels, times) -> torch.Tensor:
    """
    this is run on all time steps of the cascade N (N > K)
    """
    log_survival = -lambdas * times
    # inner_cost = -lambdas * times + torch.log(lambdas)
    inner_cost = torch.log(lambdas)
    # final_matrix = (1 - labels) * log_survival + inner_cost * labels
    final_matrix = log_survival + inner_cost * labels
    loss = -torch.sum(final_matrix)
    return loss

def loss_function_rayleigh(sigmas, k, labels, times) -> torch.Tensor:
    """
    this is run on all time steps of the cascade N (N > K)
    """
    log_survival = -torch.pow(times, 2) / (2 * torch.pow(sigmas, 2))  # TODO: Broadcasting Checked? Checked:TRUE
    # inner_cost = torch.log(times / torch.pow(sigmas, 2)) - (torch.pow(times, 2) / 2 * torch.pow(sigmas, 2))
    inner_cost = torch.log(times / torch.pow(sigmas, 2))
    # final_matrix = (1 - labels) * log_survival + inner_cost * labels
    final_matrix = log_survival + inner_cost * labels
    loss = -torch.sum(final_matrix)
    return loss
    


# calculat MAX_LEN_CASCADE
def get_max_len_cascade(cascades):
    max_len = 0
    for cascade in cascades:
        if len(cascade) > max_len:
            max_len = len(cascade)
    return max_len


def train(optimizer, model, epochs, cascades, peak_labels, device, data_loader, max_len_cascade):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (cascades, peak_labels) in enumerate(data_loader):
            cascades = cascades.to(device)
            peak_labels = peak_labels.to(device)
            hidden = model.init_hidden(cascades.shape[0], device)
            optimizer.zero_grad()
            lambdas, k = model(cascades, hidden)
            times = torch.arange(1, max_len_cascade + 1).float().to(device)
            loss = loss_function(lambdas, k, peak_labels, times)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("Epoch: {}, Loss: {}".format(epoch, total_loss))


def to_weibull_survival_function(lambdas, k, times):
    return np.exp(-np.power(times / lambdas, k))


# plot 5 random cascades
def plot_cascades(cascades, lambdas, k, peak_labels, times, num=5):
    for i in range(num):
        # plt.plot(times, cascades[i, :], label="Cascade")
        plt.plot(times, to_weibull_survival_function(lambdas[i], k[i], times), label="Weibull")
        plt.show()
        # plt.plot(times, peak_labels[i], label="Peak")
        # plt.legend()
        # plt.show()


# Dataloader and TensorDataset
def get_dataloader(cascades, peak_labels, batch_size):
    cascades = torch.from_numpy(cascades).float()
    peak_labels = torch.from_numpy(peak_labels).float()
    dataset = TensorDataset(cascades, peak_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

