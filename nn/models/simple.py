import torch
import torch.nn as nn
import torch.nn.functional as F

# Additional information
EPOCHS = 300
LOSS = 0.4
LR = 0.01
MOMENTUM = 0.9

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(12, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    from tqdm import tqdm
    import pandas as pd
    from nn.models.common import create_data_loaders, train_one_epoch, test

    net = Net()

    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM)
    #loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = nn.MSELoss()

    df = pd.read_csv('../../input/CINTE24-25_Proj1_SampleData.csv')
    train_loader, test_loader = create_data_loaders(df)

    for epoch in tqdm(range(EPOCHS)):
        train_one_epoch(model=net, optimizer=optimizer, loss_fn=loss_fn, train_loader=train_loader)
        #print(f"epoch: {epoch}/{EPOCHS}")
    test(model=net, test_loader=test_loader)
