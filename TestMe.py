from nn.models.simple_lightning import Net
# parse input file argument

# load FS model

# load NN model
model = Net.load_from_checkpoint("./nn/models/lightning_logs/simple_model/version_3/epoch=99-step=600.ckpt")

# disable randomness, dropout, etc...
model.eval()

# predict with the model
y_hat = model(x)

# Output TestResult.csv file
