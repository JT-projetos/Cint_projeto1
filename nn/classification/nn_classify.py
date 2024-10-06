import torch
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from nn.models.simple_lightning import Net
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import os


def classify(num) -> str:
    """Classify as {'Decrease', 'Increase', 'Maintain'} based on NN output (CLP)"""
    if -1 <= num <= 0.3:
        return 'Decrease'
    elif 0.3 < num <= 0.5:
        return 'Maintain'
    elif 0.5 < num <= 1:
        return 'Increase'
    else:
        raise ValueError("CLPVariation should be [-1, 1]")


DO_LABELS = False

input_file = '../../gen_input/uniform100000_class.csv'
output_file = './uniform100000_class_performance.csv'
required_columns = ('MemoryUsage', 'ProcessorLoad', 'InpNetThroughput', 'OutNetThroughput', 'OutBandwidth', 'Latency',
                    'V_MemoryUsage', 'V_ProcessorLoad', 'V_InpNetThroughput', 'V_OutNetThroughput', 'V_OutBandwidth', 'V_Latency')


if not os.path.exists(output_file) or DO_LABELS:
    df = pd.read_csv(input_file)

    # load NN model
    model = Net.load_from_checkpoint("../final_model/simple-epoch=85-val_loss=0.02.ckpt")

    # disable randomness, dropout, etc...
    model.eval()

    # predict with the model
    nn_results = model(torch.Tensor(df[list(required_columns)].values)).detach().numpy().flatten()

    df.drop(list(required_columns), axis='columns', inplace=True)

    df['nn_results'] = nn_results
    df['nn_label'] = df['CLPVariation'].apply(classify)
    #print(df[['fs_label', 'nn_label']].head())

    df[['fs_label', 'nn_label']].to_csv(output_file, index=False)
else:
    df = pd.read_csv(output_file)


conf_matrix = confusion_matrix(df['fs_label'], df['nn_label'])
# Change figure size and increase dpi for better resolution
plt.figure(figsize=(8, 7), dpi=100)
# Scale up the size of all text
sns.set(font_scale=1.1)

ax = sns.heatmap(conf_matrix, annot=True, fmt='d', )
# set x-axis label and ticks.
ax.set_xlabel("Neural Network Label", fontsize=14, labelpad=20)
ax.xaxis.set_ticklabels(df['fs_label'].unique())

# set y-axis label and ticks
ax.set_ylabel("Fuzzy System Label", fontsize=14, labelpad=20)
ax.yaxis.set_ticklabels(df['fs_label'].unique())

# set plot title
ax.set_title("Confusion Matrix for the Diabetes Detection Model", fontsize=14, pad=20)
plt.show()

print(f"{classification_report(df['fs_label'], df['nn_label'])}")
