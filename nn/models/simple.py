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
    import pandas as pd
    from nn.models.common import df_to_data_loader, train_one_epoch, test
    from sklearn.model_selection import train_test_split

    net = Net()

    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM)
    #loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = nn.MSELoss()

    df = pd.read_csv('../../input/CINTE24-25_Proj1_SampleData.csv')

    train, test = train_test_split(df, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)

    train_loader, val_loader, test_loader = df_to_data_loader(train), df_to_data_loader(val), df_to_data_loader(test)

    for epoch in range(EPOCHS):
        train_one_epoch(model=net, optimizer=optimizer, loss_fn=loss_fn, train_loader=train_loader)
        validate(model=net, loss_fn=loss_fn, val_loader=val_loader)

        #print(f"epoch: {epoch}/{EPOCHS}")
    test(model=net, test_loader=test_loader)
