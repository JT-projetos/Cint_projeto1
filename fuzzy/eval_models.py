import os
import numpy as np
import pandas as pd


# import all models
# from fuzzy.models.mamdani_gaussian import FS as FS1
# from fuzzy.models.mamdani_triangle import FS as FS2
from fuzzy.models.mamdani_triangle_v2 import FS as FS12
from fuzzy.models.mamdani_triangle_v3 import FS as FS13
#from fuzzy.models.mamdani_bell_v2 import FS as FS4
#from fuzzy.models.mamdani_bell_v5 import FS as FS5
#from fuzzy.models.mamdani_bell_v6 import FS as FS6
#from fuzzy.models.mamdani_bell_v7 import FS as FS7
#from fuzzy.models.mamdani_bell_v8 import FS as FS8
from fuzzy.models.mamdani_bell_v9 import FS as FS9
from fuzzy.models.mamdani_bell_v9a import FS as FS9a
from fuzzy.models.mamdani_bell_v9b import FS as FS9b
from fuzzy.models.mamdani_bell_v9c import FS as FS9c
#from fuzzy.models.mamdani_bell_v10 import FS as FS10


models = {
    #'mamdani_gaussian': FS1,
    #'mamdani_triangle': FS2,
    #'mamdani_triangle_v2': FS12,
    #'mamdani_triangle_v3': FS13,
    #'mamdani_bell_v2': FS4,
    #'mamdani_bell_v5': FS5,
    #'mamdani_bell_v6': FS6,
    #'mamdani_bell_v7': FS7,
    #'mamdani_bell_v8': FS8,
    'mamdani_bell_v9': FS9,
    'mamdani_bell_v9a': FS9a,
    'mamdani_bell_v9b': FS9b,
    'mamdani_bell_v9c': FS9c,
    #'mamdani_bell_v10': FS10,
}

DO_ALL_TESTS = True

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

        SystemLoad = max(row['MemoryUsage'], row['ProcessorLoad'])
        FS.set_variable("SystemLoad", SystemLoad)

        results.append(FS.inference()['CLP'])

    return results


if not os.path.exists('./output/eval_models'):
    os.mkdir('./output/eval_models')

df_test = pd.read_csv('../input/CINTE24-25_Proj1_SampleData.csv')


if 'model_results.csv' not in os.listdir('./output/eval_models/') or DO_ALL_TESTS:

    model_results = {}
    for name, model in models.items():
        model_results[name] = test_fuzzy_system(df_test, model)
    df = pd.DataFrame(model_results)
    df['CLPVariation'] = df_test['CLPVariation']
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

if __name__ == '__main__':
    from fuzzy.visualization import plot_model_scores
    plot_model_scores('./output/eval_models/model_scores.csv')
