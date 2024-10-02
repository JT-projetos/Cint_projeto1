import os
import numpy as np
import pandas as pd


# import all models
from fuzzy.models.mamdani_gaussian import FS as FS1
from fuzzy.models.mamdani_triangle import FS as FS2

models = {
    'mamdani_gaussian': FS1,
    'mamdani_triangle': FS2,
}


def test_fuzzy_system(df, FS):
    results = []
    for i, row in df.iterrows():  # FIXME get correct variable names
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
        results.append(FS.inference()['CLP'])

    return results


if not os.path.exists('./output/eval_models'):
    os.mkdir('./output/eval_models')

df_test = pd.read_csv('../input/CINTE24-25_Proj1_SampleData.csv')


if 'model_results.csv' not in os.listdir('./output/eval_models/'):

    model_results = {}
    for name, model in models.items():
        model_results[name] = test_fuzzy_system(df_test, model)
    df = pd.DataFrame(model_results)
    df.to_csv('./output/eval_models/model_results.csv', index=False)
else:
    df = pd.read_csv('./output/eval_models/model_results.csv')

#
# scores = {
#     f'datapoint {d}': [] for d in range(len(df_test))
# }
# scores['model'] = []

scores = pd.DataFrame()
for name in models.keys():
    scores[name] = np.abs((df_test['CLPVariation'] - df[name]) / df_test['CLPVariation']) * 100

    # for i, point in enumerate(s):
    #     scores['model'].append(name)
    #     scores[f'datapoint {i}'].append(point)

scores.to_csv('./output/eval_models/model_scores.csv', index=False)
