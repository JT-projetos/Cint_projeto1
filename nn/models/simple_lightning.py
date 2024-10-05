import lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import pandas as pd


class Net(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(12, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat.squeeze(), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        val_loss = F.mse_loss(y_hat, y.unsqueeze(1))
        self.log("val_loss", val_loss)
        #self.log("val_acc", 100)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.mse_loss(y_hat, y.unsqueeze(1))
        self.log("test_loss", test_loss)


def df_to_tensor(df, target_column='CLPVariation'):
    # Creating np arrays
    data_target = df[target_column].values
    data_features = df.drop(target_column, axis='columns').values

    # Passing to DataLoader
    data_tensor = data_utils.TensorDataset(torch.Tensor(data_features), torch.Tensor(data_target))
    return data_tensor


if __name__ == '__main__':
    # -------------------
    # Step 2: Define data
    # -------------------
    df = pd.read_csv('../../gen_input/.csv')

    train, test = train_test_split(df, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)

    train, val, test = df_to_tensor(train), df_to_tensor(val), df_to_tensor(test)

    # -------------------
    # Step 3: Train
    # -------------------

    logger = TensorBoardLogger('../model_logs', name='simple_model')
    checkpoints = ModelCheckpoint(monitor='val_loss', mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', mode='min')

    model = Net()
    trainer = L.Trainer(max_epochs=100, logger=logger, callbacks=[checkpoints, early_stopping])
    trainer.fit(model, data_utils.DataLoader(train), data_utils.DataLoader(val))
    trainer.test(model, data_utils.DataLoader(test))
