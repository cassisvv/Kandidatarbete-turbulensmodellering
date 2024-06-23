import torch
import torch.nn as nn


#-----------------Neural Network setup----------------------------------------------

class ThePredictionMachine(nn.Module):

    def __init__(self, input, output, hidden, nodes, bnorm=False):
        super(ThePredictionMachine, self).__init__()

        layers = []

        # Add input layer
        layers.append(nn.Linear(input, nodes))
        if bnorm == True:
            layers.append(nn.BatchNorm1d(nodes))
        layers.append(nn.ReLU())

        # Add hidden layers
        for _ in range(hidden - 1):
            layers.append(nn.Linear(nodes, nodes))
            if bnorm == True:
                layers.append(nn.BatchNorm1d(nodes))
            layers.append(nn.ReLU())

        # Add output layer
        layers.append(nn.Linear(nodes, output))

        # Combine all layers into a sequential module
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    print('in train_loop: len(dataloader)', len(dataloader))
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        #       optimizer.zero_grad()
        # https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
        optimizer.zero_grad(None)
        loss.backward()
        optimizer.step()


def test_loop(dataloader, model, loss_fn):
    global pred_numpy, pred1, size1
    size = len(dataloader.dataset)
    size1 = size
    num_batches = len(dataloader)
    test_loss = 0
    print('in test_loop: len(dataloader)', len(dataloader))

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # transform from tensor to numpy
            pred_numpy = pred.detach().numpy()

    test_loss /= num_batches

    print(f"Avg loss: {test_loss:>.2e} \n")

    return test_loss
