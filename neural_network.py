'''

This module is used to train the networks.
The network structure has to be changed manually and also has to match with the global parameters width, depth, to make
sure the trained models are correctly saved.

Per training session, 4 model state dictionaries are saved (one for each fold), 4 network loss plots are saved (one for each fold).

'''

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold


# Get either GPU or CPU device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

width = 100
depth = 2


class NeuralNetwork(nn.Module):

    def __init__(self):

        super(NeuralNetwork, self).__init__()

        self.flatten = nn.Flatten()

        # Change model here
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(7, width),
            # nn.ReLU(),
            # nn.Linear(width, width),
            # nn.ReLU(),
            # nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        output = self.linear_relu_stack(x)
        return output

def get_model():

    model = NeuralNetwork()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    return model, optimizer, loss_fn


def transform_data(X_train, X_val, y_train, y_val, batch_size):

    X_train, X_val, y_train, y_val = map(torch.Tensor, (X_train, X_val, y_train, y_val))

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size*2)

    return train_dl, val_dl


def loss_batch(model, loss_func, X_b, y_b, opt=None):

    loss = loss_func(model(X_b).flatten(), y_b)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(X_b)


def reset_network(model):

    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def fit(epochs, model, loss_fn, optimizer, train_dl, val_dl, fold, batch_size, verbose=True):

    model.train()

    val_losses = []
    train_losses = []

    for e in range(epochs):

        t_losses, t_bs = zip(*[loss_batch(model, loss_fn, X_b, y_b, optimizer) for X_b, y_b in train_dl])
        train_loss = np.sum(np.multiply(t_losses, t_bs)) / np.sum(t_bs)
        train_losses.append(train_loss)

        model.eval()

        with torch.no_grad():
            v_losses, v_bs = zip(*[loss_batch(model, loss_fn, X_b, y_b) for X_b, y_b in val_dl])

        val_loss = np.sum(np.multiply(v_losses, v_bs)) / np.sum(v_bs)
        val_losses.append(val_loss)

        if verbose:
            print(e, val_loss)

    torch.save(model.state_dict(), 'models/relu_' + depth * f'_{width}' + f'batch_size_{batch_size}_fold_{fold}.pt')
    plot_network_behaviour(train_losses, val_losses, epochs, batch_size, fold)


def plot_network_behaviour(train_losses, val_losses, epochs, batch_size, fold):

    plt.plot(range(epochs), train_losses, label='Training Loss')
    plt.plot(range(epochs), val_losses, label='Validation Loss')
    plt.title('Model Loss Curves, Network: ReLU' + depth * f'_{width}')  # Change model here
    plt.xlabel('epochs')
    plt.ylabel('L1-Loss')
    plt.legend()
    plt.savefig(f'plots/relu_' + depth * f'_{width}' + f'batch_size_{batch_size}_fold_{fold}.png')  # Change model here
    plt.close()


if __name__ == '__main__':

    X = np.load(f'data/training/in_5_20_20.npz')['arr_0']
    y = np.load(f'data/training/ts_5_20_20.npz')['arr_0']

    epochs = 100
    batch_size = 125

    kfold = KFold(n_splits=4, shuffle=True, random_state=42)

    for fold, (train_ids, val_ids) in enumerate(kfold.split(X)):

        print(f'-------------------- Fold: {fold+1} -------------------- \n')

        X_train, X_val = X[train_ids], X[val_ids]
        y_train, y_val = y[train_ids], y[val_ids]

        train_dl, val_dl = transform_data(X_train, X_val, y_train, y_val, batch_size)

        model, optimizer, loss_fn = get_model()
        fit(epochs, model, loss_fn, optimizer, train_dl, val_dl, fold+1, batch_size, verbose=True)
        reset_network(model)
















