import argparse
import pandas as pd
from nn.models.simple_lightning import Net
import torch
import os
import pickle


parser = argparse.ArgumentParser(
                    prog='TestMe -> Fuzzy & Neural Network CInt',
                    description='Tests a Fuzzy Inference System and a Neural Network on a given .csv file',
                    epilog='Thank you for choosing us!')

parser.add_argument('filename', required=True)
args = parser.parse_args()

# parse input file argument
df = pd.read_csv(args.filename)

required_columns = ('MemoryUsage', 'ProcessorLoad', 'InpNetThroughput', 'OutNetThroughput', 'OutBandwidth', 'Latency',
                    'V_MemoryUsage', 'V_ProcessorLoad', 'V_InpNetThroughput', 'V_OutNetThroughput', 'V_OutBandwidth', 'V_Latency')

if not all(e in required_columns for e in df.columns):
    raise RuntimeError('Required columns are missing')

# load FS model
with open('./final_models/fs.pkl', 'rb') as f:
    FS = pickle.load(f)

fs_results = FS.predict(df)

del FS

# load NN model
model = Net.load_from_checkpoint("./final_models/nn.ckpt")

# disable randomness, dropout, etc...
model.eval()

# predict with the model
nn_results = model(torch.Tensor(df.values)).detach().numpy()

# Output TestResult.csv file
df = pd.DataFrame({
    'CLPVariation_FS': fs_results,
    'CLPVariation_NN': nn_results,
})
df.to_csv('TestResult.csv', index=False)
