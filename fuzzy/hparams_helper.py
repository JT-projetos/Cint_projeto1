
mamdani_bellv9_best_params = {
    'system_load_very_low_a_bell': 4,
    'system_load_very_low_b_bell': 0.5,
    'system_load_very_low_c_bell': 0,
    'system_load_low_a': 4,
    'system_load_low_b': 0.05,
    'system_load_low_c': 0.5,
    'system_load_medium_a': 2,
    'system_load_medium_b': 0.1,
    'system_load_medium_c': 0.65,
    'system_load_high_a': 2,
    'system_load_high_b': 0.05,
    'system_load_high_c': 0.8,
    'system_load_very_high_a_tri': 0.8,
    'latency_low_a': 4,
    'latency_low_b': 0.15,
    'latency_low_c': 0.1,
    'latency_medium_a': 4,
    'latency_medium_b': 0.15,
    'latency_medium_c': 0.45,
    'latency_high_a': 4,
    'latency_high_b': 0.15,
    'latency_high_c': 0.9,
    'clp_very_low_c_tri': -0.8,
    'clp_low_a': 2,
    'clp_low_b': 0.05,
    'clp_low_c': -0.65,
    'clp_medium_a': 1,
    'clp_medium_b': 0.1,
    'clp_medium_c': 0.1,
    'clp_high_a': 2,
    'clp_high_b': 0.05,
    'clp_high_c': 0.5,
    'clp_very_high_a_tri': 0.7,
}


def convert_optuna_to_hparams(p: dict) -> dict:
    return {
        'SystemLoad': {
            'universe_of_discourse': [0, 1],
            'linguistic_variables': [
                {
                    'function': 'bell_mf',
                    'term': 'very_low',
                    'params': {'a': p['system_load_very_low_a_bell'], 'b': p['system_load_very_low_b_bell'],
                               'c': p['system_load_very_low_c_bell']},
                },
                {
                    'function': 'bell_mf',
                    'term': 'low',
                    'params': {'a': p['system_load_low_a'], 'b': p['system_load_low_b'], 'c': p['system_load_low_c']},
                },
                {
                    'function': 'bell_mf',
                    'term': 'medium',
                    'params': {'a': p['system_load_medium_a'], 'b': p['system_load_medium_b'], 'c': p['system_load_medium_c']},
                },
                {
                    'function': 'bell_mf',
                    'term': 'high',
                    'params': {'a': p['system_load_high_a'], 'b': p['system_load_high_b'], 'c': p['system_load_high_c']},
                },
                {
                    'function': 'triangle_mf',
                    'term': 'very_high',
                    'params': {'a': 0.8, 'b': 1, 'c': 1},
                },
                {
                    'function': 'triangle_mf',
                    'term': 'very_high',
                    'params': {'a': p['system_load_very_high_a_tri'], 'b': 1, 'c': 1},
                },
            ],
        },
        'Latency': {
            'universe_of_discourse': [0, 1],
            'linguistic_variables': [
                {
                    'function': 'bell_mf',
                    'term': 'low',
                    'params': {'a': p['latency_low_a'], 'b': p['latency_low_b'], 'c': p['latency_low_c']},
                },
                {
                    'function': 'bell_mf',
                    'term': 'medium',
                    'params': {'a': p['latency_medium_a'], 'b': p['latency_medium_b'], 'c': p['latency_medium_c']},
                },
                {
                    'function': 'bell_mf',
                    'term': 'high',
                    'params': {'a': p['latency_high_a'], 'b': p['latency_high_b'], 'c': p['latency_high_c']},
                },
            ],
        },
        'CLP': {
            'universe_of_discourse': [-1, 1],
            'linguistic_variables': [
                {
                    'function': 'triangle_mf',
                    'term': 'decrease_significantly',
                    'params': {'a': -1, 'b': -1, 'c': p['clp_very_low_c_tri']},
                },
                {
                    'function': 'bell_mf',
                    'term': 'decrease',
                    'params': {'a': p['clp_low_a'], 'b': p['clp_low_b'], 'c': p['clp_low_c']},
                },
                {
                    'function': 'bell_mf',
                    'term': 'maintain',
                    'params': {'a': p['clp_medium_a'], 'b': p['clp_medium_b'], 'c': p['clp_medium_c']},
                },
                {
                    'function': 'bell_mf',
                    'term': 'increase',
                    'params': {'a': p['clp_high_a'], 'b': p['clp_high_b'], 'c': p['clp_high_c']},
                },
                {
                    'function': 'triangle_mf',
                    'term': 'increase_significantly',
                    'params': {'a': p['clp_very_high_a_tri'], 'b': 1, 'c': 1},
                },
            ],
        },
    }

