import torch
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from nn.models.simple_lightning import Net
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support as score
import os

from generate_data import classify


DO_LABELS = True

input_file = '../../gen_input/trianglev4_uniform10000_class.csv'
output_file = './trianglev4_uniform10000_class_performance.csv'
required_columns = ('MemoryUsage', 'ProcessorLoad', 'InpNetThroughput', 'OutNetThroughput', 'OutBandwidth', 'Latency',
                    'V_MemoryUsage', 'V_ProcessorLoad', 'V_InpNetThroughput', 'V_OutNetThroughput', 'V_OutBandwidth', 'V_Latency')


if not os.path.exists(output_file) or DO_LABELS:
    df = pd.read_csv(input_file)

    # load NN model
    model = Net.load_from_checkpoint("../final_model/trianglev4-epoch=172-val_loss=0.03.ckpt")

    # disable randomness, dropout, etc...
    model.eval()

    # predict with the model
    nn_results = model(torch.Tensor(df[list(required_columns)].values)).detach().numpy().flatten()

    df.drop(list(required_columns), axis='columns', inplace=True)

    df['nn_results'] = nn_results
    df['nn_label'] = df['nn_results'].apply(classify)
    #print(df[['fs_label', 'nn_label']].head())

    df[['fs_label', 'nn_label']].to_csv(output_file, index=False)
else:
    df = pd.read_csv(output_file)


conf_matrix = confusion_matrix(df['fs_label'], df['nn_label'])
# Change figure size and increase dpi for better resolution
plt.figure(figsize=(8, 7), dpi=100)
# Scale up the size of all text
sns.set(font_scale=1.1)
labels=['Decrease', 'Increase', 'Maintain']
ax = sns.heatmap(conf_matrix, annot=True, fmt='d', )
# set x-axis label and ticks.
ax.set_xlabel("Neural Network Label", fontsize=14, labelpad=20)
ax.xaxis.set_ticklabels(labels)

# set y-axis label and ticks
ax.set_ylabel("Fuzzy System Label", fontsize=14, labelpad=20)
ax.yaxis.set_ticklabels(labels)

# set plot title
ax.set_title("Confusion Matrix for Neural Network Classification", fontsize=14, pad=20)
plt.show()

precision, recall, f1score, support = score(df['fs_label'], df['nn_label'])

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('f1score: {}'.format(f1score))
print('support: {}'.format(support))
#print(f"{type(precision)}, {precision}")
df = pd.DataFrame({
    'precision': precision,
    'recall': recall,
    'f1-score': f1score,
    'label': ['Decrease', 'Maintain', 'Increase']
})
print(df.to_latex(index=False, float_format='%.2f'))
