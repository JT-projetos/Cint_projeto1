from simpful import FuzzySystem, LinguisticVariable, FuzzySet, TriangleFuzzySet, Trapezoidal_MF
from fuzzy.models.bell_mf import Bell_MF
from fuzzy.fuzzy_system_wrapper import FuzzySystemWrapper

"""
mamdani_hparams MSE: 0.04633350919848102
"""

hparams = {
    'SystemLoad': {
        'universe_of_discourse': [0, 1],
        'linguistic_variables': [
            {
                'function': 'bell_mf',
                'term': 'very_low',
                'params': {'a': 4, 'b': 0.5, 'c': 0},
            },
            {
                'function': 'bell_mf',
                'term': 'low',
                'params': {'a': 4, 'b': 0.05, 'c': 0.5},
            },
            {
                'function': 'bell_mf',
                'term': 'medium',
                'params': {'a': 2, 'b': 0.1, 'c': 0.65},
            },
            {
                'function': 'bell_mf',
                'term': 'high',
                'params': {'a': 2, 'b': 0.05, 'c': 0.8},
            },
            {
                'function': 'triangle_mf',
                'term': 'very_high',
                'params': {'a': 0.8, 'b': 1, 'c': 1},
            },
        ],
    },
    'Latency': {
        'universe_of_discourse': [0, 1],
        'linguistic_variables': [
            {
                'function': 'bell_mf',
                'term': 'low',
                'params': {'a': 4, 'b': 0.15, 'c': 0.1},
            },
            {
                'function': 'bell_mf',
                'term': 'medium',
                'params': {'a': 4, 'b': 0.15, 'c': 0.45},
            },
            {
                'function': 'bell_mf',
                'term': 'high',
                'params': {'a': 4, 'b': 0.15, 'c': 0.9},
            },
        ],
    },
    'CLP': {
        'universe_of_discourse': [-1, 1],
        'linguistic_variables': [
            {
                'function': 'triangle_mf',
                'term': 'decrease_significantly',
                'params': {'a': -1, 'b': -1, 'c': -0.8},
            },
            {
                'function': 'bell_mf',
                'term': 'decrease',
                'params': {'a': 2, 'b': 0.05, 'c': -0.65},
            },
            {
                'function': 'bell_mf',
                'term': 'maintain',
                'params': {'a': 1, 'b': 0.1, 'c': 0.1},
            },
            {
                'function': 'bell_mf',
                'term': 'increase',
                'params': {'a': 2, 'b': 0.05, 'c': 0.5},
            },
            {
                'function': 'triangle_mf',
                'term': 'increase_significantly',
                'params': {'a': 0.7, 'b': 1, 'c': 1},
            },
        ],
    },
}


def create_fuzzy_set_from_dict(var: dict) -> FuzzySet:
    match var['function']:
        case 'bell_mf':
            F = FuzzySet(function=Bell_MF(**var['params']), term=var['term'])
        case 'triangle_mf':
            F = TriangleFuzzySet(**var['params'], term=var['term'])

        case 'trapezoid_mf':
            F = FuzzySet(function=Trapezoidal_MF(**var['params']), term=var['term'])

        case _ as e:
            raise RuntimeError(f"The function {e} is not supported yet")

    return F


def create_fuzzy_system(hparams: dict, rules=None) -> FuzzySystemWrapper:
    FS = FuzzySystemWrapper()

    for k in hparams.keys():
        lv = hparams[k]
        FS.add_linguistic_variable(k, LinguisticVariable(
            [create_fuzzy_set_from_dict(var) for var in lv['linguistic_variables']],
            universe_of_discourse=lv['universe_of_discourse']))

    if rules is None:
        rules = [
            "IF (SystemLoad IS low) THEN (CLP IS increase)",
            "IF (SystemLoad IS medium) THEN (CLP IS maintain)",
            "IF (SystemLoad IS high) THEN (CLP IS decrease)",
            "IF (SystemLoad IS very_low) THEN (CLP IS increase_significantly)",
            "IF (SystemLoad IS very_high) THEN (CLP IS decrease_significantly)",
            "IF (Latency IS high) AND ((SystemLoad IS very_low) OR (SystemLoad IS low) OR (SystemLoad IS medium)) THEN (CLP IS increse_significantly)",
        ]

        #rules = [
#
        #    "IF (SystemLoad IS critical) AND (OutBandwidth IS high) THEN (CLP IS decrease_significantly)",
#
        #    "IF (SystemLoad IS critical) AND (OutBandwidth IS low) THEN (CLP IS decrease)",
#
        #    "IF (SystemLoad IS critical) AND (OutBandwidth IS medium) THEN (CLP IS decrease)",
#
        #    "IF (SystemLoad IS high) AND (Latency IS high) THEN (CLP IS increase_significantly)",
#
        #    "IF (SystemLoad IS high) AND (Latency IS low) THEN (CLP IS maintain)",
#
        #    "IF (SystemLoad IS moderate) THEN (CLP IS increase)",
#
        #    "IF (SystemLoad IS low) THEN (CLP IS increase_significantly)",
#
        #    "IF (SystemLoad IS very_low) THEN (CLP IS increase_significantly)",
#
        #]
    FS.add_rules(rules)

    return FS



if __name__ == '__main__':
    import os
    import pickle
    from fuzzy.visualization import *

    save_path = '../output/mamdani_bell_hparams'

    FS = create_fuzzy_system(hparams=hparams)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plot_inputs_outputs_fuzzy_system(FS, save_path)
    # plot_memory_processor_clp(FS, save_path)

    with open(f"{save_path}/model.pkl", "wb") as f:
        pickle.dump(FS, f)
