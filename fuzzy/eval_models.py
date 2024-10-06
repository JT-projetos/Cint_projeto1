import numpy as np
import pandas as pd


# import all models
# from fuzzy.models.mamdani_gaussian import FS as FS1
# from fuzzy.models.mamdani_triangle import FS as FS2
from fuzzy.models.deprecated.mamdani_bell_v2 import FS as FS4
#from fuzzy.models.mamdani_bell_v5 import FS as FS5
#from fuzzy.models.mamdani_bell_v6 import FS as FS6
#from fuzzy.models.mamdani_bell_v7 import FS as FS7
#from fuzzy.models.mamdani_bell_v8 import FS as FS8

from fuzzy.models.deprecated.mamdani_triangle_v2 import FS as FS12
from fuzzy.models.mamdani_triangle_v3 import FS as FS13
from fuzzy.models.mamdani_triangle_v4 import FS as FS14
from fuzzy.models.mamdani_bell_v9 import FS as FS9
from fuzzy.models.mamdani_hparams import create_fuzzy_system
#from fuzzy.models.mamdani_bell_v10 import FS as FS10

import os
import json


def relative_error(y_true, y_pred):
    return np.abs((y_true - y_pred) / y_true) * 100


def mse(y_true, y_pred):
    return ((y_true - y_pred)**2)/len(y_true)


with open('./output/hparams_007.json', 'r') as f:
    #hparams = convert_optuna_to_hparams(json.load(f))
    FSbest = create_fuzzy_system(hparams=json.load(f))

models = {
    #'mamdani_gaussian': FS1,
    #'mamdani_triangle': FS2,
    #'mamdani_triangle_v2': FS12,
    #'mamdani_triangle_v3': FS13,
    'mamdani_triangle_v4': FS14,
    #'mamdani_bell_v2': FS4,
    #'mamdani_bell_v5': FS5,
    #'mamdani_bell_v6': FS6,
    #'mamdani_bell_v7': FS7,
    #'mamdani_bell_v8': FS8,
    #'mamdani_bell_v9': FS9,
    #'mamdani_bell_v9a': FS9a,
    #'mamdani_bell_v9b': FS9b,
    #'mamdani_bell_v9c': FS9c,
    #'mamdani_hparams': FShparams,
    #'mamdani_best': FSbest,
}

DO_ALL_TESTS = True

if not os.path.exists('./output/eval_models'):
    os.mkdir('./output/eval_models')

df_test = pd.read_csv('../input/CINTE24-25_Proj1_SampleData.csv')


if 'model_results.csv' not in os.listdir('./output/eval_models/') or DO_ALL_TESTS:

    model_results = {}
    for name, model in models.items():
        model_results[name] = model.predict(df_test)
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
MSE = {}
scores = pd.DataFrame()
for name in models.keys():
    scores[name] = mse(y_true=df_test['CLPVariation'], y_pred=df[name])
    MSE[name] = (((df_test['CLPVariation'] - df[name])**2)/len(df_test['CLPVariation'])).sum()

    # for i, point in enumerate(s):
    #     scores['model'].append(name)
    #     scores[f'datapoint {i}'].append(point)

for name in models.keys():
    print(f"{name} MSE: {MSE[name]}")

scores.to_csv('./output/eval_models/model_scores.csv', index=False)

if __name__ == '__main__':
    from fuzzy.visualization import plot_model_scores
    plot_model_scores('./output/eval_models/model_scores.csv')
    
