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

fs_results = []
# load FS model
with open('./final_models/fs.pkl', 'rb') as f:
    FS = pickle.load(f)

# Maybe in future create a predict function common to all FS that receives df
for row in df.iterrows():  # FIXME get correct variable names
    FS.set_variable('MemoryUsage', row['MemoryUsage'])
    FS.set_variable('ProcessorLoad', row['ProcessorLoad'])
    FS.set_variable('InpNetThroughput', row['InpNetThroughput'])
    FS.set_variable('OutNetThroughput', row['OutNetThroughput'])
    FS.set_variable('OutBandwidth', row['OutBandwidth'])
    FS.set_variable('Latency', row['Latency'])
    FS.set_variable('V_MemoryUsage', row['V_MemoryUsage'])
    FS.set_variable('V_ProcessorLoad', row['V_ProcessorLoad'])
    FS.set_variable('V_InpNetThroughput', row['V_InpNetThroughput'])
    FS.set_variable('V_OutNetThroughput', row['V_OutNetThroughput'])
    FS.set_variable('V_OutBandwidth', row['V_OutBandwidth'])
    FS.set_variable('V_Latency', row['V_Latency'])
    fs_results.append(FS.inference()['CLP'])

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
