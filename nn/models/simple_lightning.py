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
    df = pd.read_csv('../../gen_input/uniform10000.csv')
    #df = pd.read_csv('../../input/CINTE24-25_Proj1_SampleData.csv')

    train, test = train_test_split(df, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)

    train, val, test = df_to_tensor(train), df_to_tensor(val), df_to_tensor(test)

    # -------------------
    # Step 3: Train
    # -------------------

    logger = TensorBoardLogger('../model_logs', name='second_model')
    checkpoint = ModelCheckpoint(monitor='val_loss', mode='min', filename='simple-{epoch:02d}-{val_loss:.2f}')
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=5)

    #model = Net()
    model = Net.load_from_checkpoint('../model_logs/second_model/version_0/checkpoints/simple-epoch=168-val_loss=0.03.ckpt')
    trainer = L.Trainer(max_epochs=300, logger=logger, callbacks=[checkpoint, early_stopping])
    try:
        trainer.fit(model, data_utils.DataLoader(train), data_utils.DataLoader(val))
    except KeyboardInterrupt:
        pass
    trainer.test(model, data_utils.DataLoader(test))
