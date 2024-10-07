import argparse

import numpy as np
import pandas as pd
from nn.models.simple_lightning import Net
import torch
from fuzzy.models.mamdani_triangle_v4 import FS
from nn.classification.generate_data import classify


def mse(y_true, y_pred):
    return ((y_true - y_pred)**2)/len(y_true)


SHOW_LATEX = False
SHOW_BAR_PLOT = True
SHOW_CONF_MATRIX = True

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
model = Net.load_from_checkpoint("nn/final_model/trianglev4-epoch=172-val_loss=0.03.ckpt")

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
    print("Test Results")
    print(df_result.to_latex(index=False, float_format='%.4f'))

if SHOW_BAR_PLOT:
    from fuzzy.visualization import plot_model_scores
    plot_model_scores(df_result[['MSE_FS', 'MSE_NN']])

if SHOW_CONF_MATRIX:
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support as score
    import seaborn as sns
    from matplotlib import pyplot as plt

    df = df_result
    conf_matrix = confusion_matrix(df['CLP_label_FS'], df['CLP_label_NN'])
    # Change figure size and increase dpi for better resolution
    plt.figure(figsize=(8, 7), dpi=100)
    # Scale up the size of all text
    sns.set(font_scale=1.1)

    ax = sns.heatmap(conf_matrix, annot=True, fmt='d', )
    # set x-axis label and ticks.
    ax.set_xlabel("Neural Network Label", fontsize=14, labelpad=20)
    ax.xaxis.set_ticklabels(df['CLP_label_FS'].unique())

    # set y-axis label and ticks
    ax.set_ylabel("Fuzzy System Label", fontsize=14, labelpad=20)
    ax.yaxis.set_ticklabels(df['CLP_label_FS'].unique())

    # set plot title
    ax.set_title("Confusion Matrix for Neural Network Classification", fontsize=14, pad=20)
    plt.show()

    precision, recall, f1score, support = score(df['CLP_label_FS'], df['CLP_label_NN'])

    df = pd.DataFrame({
        'precision': precision,
        'recall': recall,
        'f1-score': f1score,
        'label': ['Decrease', 'Maintain', 'Increase']
    })
    print("Precision Recall F1-Score Table")
    print(df.to_latex(index=False, float_format='%.2f'))

