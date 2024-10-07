import argparse

import numpy as np
import pandas as pd
from nn.models.simple_lightning import Net
import torch
from fuzzy.models.mamdani_bell_hyper import FS
from nn.classification.nn_classify import classify

def mse(y_true, y_pred):
    return ((y_true - y_pred)**2)/len(y_true)


SHOW_LATEX = True
SHOW_BAR_PLOT = True

parser = argparse.ArgumentParser(
                    prog='TestMe -> Fuzzy & Neural Network CInt',
                    description='Tests a Fuzzy Inference System and a Neural Network on a given .csv file',
                    epilog='Thank you for choosing us!')

parser.add_argument('filename', type=str, help="Filename of the .csv file")
args = parser.parse_args()


# parse input file argument
df = pd.read_csv(args.filename)

required_columns = ('MemoryUsage', 'ProcessorLoad', 'InpNetThroughput', 'OutNetThroughput', 'OutBandwidth', 'Latency',
                    'V_MemoryUsage', 'V_ProcessorLoad', 'V_InpNetThroughput', 'V_OutNetThroughput', 'V_OutBandwidth', 'V_Latency')

if not all(r in df.columns for r in required_columns):
    raise RuntimeError('Required columns are missing')

# only use the required columns
#df = df[list(required_columns)]  # this is required because NN model needs 12 features as input

# test FS model
fs_results = FS.predict(df[list(required_columns)])


# load NN model
model = Net.load_from_checkpoint("nn/final_model/simple-epoch=85-val_loss=0.02.ckpt")

# disable randomness, dropout, etc...
model.eval()

# predict with the model
nn_results = model(torch.Tensor(df[list(required_columns)].values)).detach().numpy().flatten()

df_results = {
    'CLPVariation_FS': fs_results,
    'CLPVariation_NN': nn_results,

    'CLP_label_FS': np.array([classify(x) for x in fs_results]),
    'CLP_label_NN': np.array([classify(x) for x in nn_results]),
}

if 'CLPVariation' in df.columns:
    df_results['MSE_FS'] = mse(df['CLPVariation'], fs_results)
    df_results['MSE_NN'] = mse(df['CLPVariation'], nn_results)

# Output TestResult.csv file
df_result = pd.DataFrame(df_results)
df_result.to_csv('TestResult.csv', index=False)
print(df_result.head(10))
if SHOW_LATEX:
    print(df_result.to_latex(index=False, float_format='%.3f'))

if SHOW_BAR_PLOT:
    from fuzzy.visualization import plot_model_scores
    plot_model_scores(df_result[['MSE_FS', 'MSE_NN']])
