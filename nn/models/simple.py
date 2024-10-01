import torch
import torch.nn as nn
import torch.nn.functional as F

# Additional information
EPOCHS = 5
LOSS = 0.4
LR = 0.01
MOMENTUM = 0.9

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(12, 32),
            nn.Sigmoid(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


def train_one_epoch(model, optimizer, loss_fn, train_loader):
    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

if __name__ == '__main__':


    net = Net()

    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM)
    #loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = nn.MSELoss()

    for epoch in EPOCHS:
        train_one_epoch(model=net, optimizer=optimizer, loss_fn=loss_fn, train_loader=)

