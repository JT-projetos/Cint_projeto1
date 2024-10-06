import optuna
# from optuna import trial
import optuna.visualization as vis
import pandas as pd
from fuzzy.models.mamdani_hparams import create_fuzzy_system
from fuzzy.hparams_helper import convert_optuna_to_hparams, mamdani_bellv9_best_params, mamdani_triangle_best_params, \
    convert_optuna_to_hparams_tri
import os

def sample_triangular_hparams(trial: optuna.Trial):
    # SystemLoad Hyperparameters
    system_load_very_low_a = trial.suggest_float("system_load_very_low_a", 0, 1)
    system_load_very_low_b = trial.suggest_float("system_load_very_low_b", system_load_very_low_a,1)
    system_load_very_low_c = trial.suggest_float("system_load_very_low_c", system_load_very_low_b, 1)

    system_load_low_a = trial.suggest_float("system_load_low_a", 0, 1)
    system_load_low_b = trial.suggest_float("system_load_low_b", system_load_low_a, 1)
    system_load_low_c = trial.suggest_float("system_load_low_c", system_load_low_b, 1)

    system_load_moderate_a = trial.suggest_float("system_load_moderate_a", 0, 1)
    system_load_moderate_b = trial.suggest_float("system_load_moderate_b", system_load_moderate_a, 1)
    system_load_moderate_c = trial.suggest_float("system_load_moderate_c", system_load_moderate_b, 1)

    system_load_high_a = trial.suggest_float("system_load_high_a", 0, 1)
    system_load_high_b = trial.suggest_float("system_load_high_b", system_load_high_a, 1)
    system_load_high_c = trial.suggest_float("system_load_high_c", system_load_high_b, 1)

    system_load_critical_a = trial.suggest_float("system_load_critical_a", 0, 1)
    system_load_critical_b = trial.suggest_float("system_load_critical_b", system_load_critical_a, 1)
    system_load_critical_c = trial.suggest_float("system_load_critical_c", system_load_critical_b, 1)

    # Latency Hyperparameters
    latency_low_a = trial.suggest_float("latency_low_a", 0, 1)
    latency_low_b = trial.suggest_float("latency_low_b", latency_low_a, 1)
    latency_low_c = trial.suggest_float("latency_low_c", latency_low_b, 1)

    latency_moderate_a = trial.suggest_float("latency_moderate_a", 0, 1)
    latency_moderate_b = trial.suggest_float("latency_moderate_b", latency_moderate_a, 1)
    latency_moderate_c = trial.suggest_float("latency_moderate_c", latency_moderate_b, 1)

    latency_high_a = trial.suggest_float("latency_high_a", 0, 1)
    latency_high_b = trial.suggest_float("latency_high_b", latency_high_a, 1)
    latency_high_c = trial.suggest_float("latency_high_c", latency_high_b, 1)

    # OutBandwidth Hyperparameters
    out_bandwidth_low_a = trial.suggest_float("out_bandwidth_low_a", 0, 1)
    out_bandwidth_low_b = trial.suggest_float("out_bandwidth_low_b", out_bandwidth_low_a, 1)
    out_bandwidth_low_c = trial.suggest_float("out_bandwidth_low_c", out_bandwidth_low_b, 1)

    out_bandwidth_medium_a = trial.suggest_float("out_bandwidth_medium_a", 0, 1)
    out_bandwidth_medium_b = trial.suggest_float("out_bandwidth_medium_b", out_bandwidth_medium_a, 1)
    out_bandwidth_medium_c = trial.suggest_float("out_bandwidth_medium_c", out_bandwidth_medium_b, 1)

    out_bandwidth_high_a = trial.suggest_float("out_bandwidth_high_a", 0, 1)
    out_bandwidth_high_b = trial.suggest_float("out_bandwidth_high_b", out_bandwidth_high_a, 1)
    out_bandwidth_high_c = trial.suggest_float("out_bandwidth_high_c", out_bandwidth_high_b, 1)
    out_bandwidth_high_d = trial.suggest_float("out_bandwidth_high_d", out_bandwidth_high_c, 1)

    # CLP Hyperparameters
    clp_decrease_significantly_a = trial.suggest_float("clp_decrease_significantly_a", -1, 1)
    clp_decrease_significantly_b = trial.suggest_float("clp_decrease_significantly_b", clp_decrease_significantly_a, 1)
    clp_decrease_significantly_c = trial.suggest_float("clp_decrease_significantly_c", clp_decrease_significantly_b, 1)

    clp_decrease_a = trial.suggest_float("clp_decrease_a", -1, 1)
    clp_decrease_b = trial.suggest_float("clp_decrease_b", clp_decrease_a, 1)
    clp_decrease_c = trial.suggest_float("clp_decrease_c", clp_decrease_b, 1)

    clp_maintain_a = trial.suggest_float("clp_maintain_a", -1, 1)
    clp_maintain_b = trial.suggest_float("clp_maintain_b",  clp_maintain_a, 1)
    clp_maintain_c = trial.suggest_float("clp_maintain_c", clp_maintain_b, 1)

    clp_increase_a = trial.suggest_float("clp_increase_a", -1, 1)
    clp_increase_b = trial.suggest_float("clp_increase_b", clp_increase_a , 1)
    clp_increase_c = trial.suggest_float("clp_increase_c", clp_increase_b, 1)

    clp_increase_significantly_a = trial.suggest_float("clp_increase_significantly_a", -1, 1)
    clp_increase_significantly_b = trial.suggest_float("clp_increase_significantly_b", clp_increase_significantly_a, 1)
    clp_increase_significantly_c = trial.suggest_float("clp_increase_significantly_c", clp_increase_significantly_b, 1)

    return {
        'SystemLoad': {
            'universe_of_discourse': [0, 1],
            'linguistic_variables': [
                {'function': 'triangle_mf', 'term': 'very_low', 'params': {'a': system_load_very_low_a, 'b': system_load_very_low_b, 'c': system_load_very_low_c}},
                {'function': 'triangle_mf', 'term': 'low', 'params': {'a': system_load_low_a, 'b': system_load_low_b, 'c': system_load_low_c}},
                {'function': 'triangle_mf', 'term': 'moderate', 'params': {'a': system_load_moderate_a, 'b': system_load_moderate_b, 'c': system_load_moderate_c}},
                {'function': 'triangle_mf', 'term': 'high', 'params': {'a': system_load_high_a, 'b': system_load_high_b, 'c': system_load_high_c}},
                {'function': 'triangle_mf', 'term': 'critical', 'params': {'a': system_load_critical_a, 'b': system_load_critical_b, 'c': system_load_critical_c}},
            ],
        },
        'Latency': {
            'universe_of_discourse': [0, 1],
            'linguistic_variables': [
                {'function': 'triangle_mf', 'term': 'low', 'params': {'a': latency_low_a, 'b': latency_low_b, 'c': latency_low_c}},
                {'function': 'triangle_mf', 'term': 'moderate', 'params': {'a': latency_moderate_a, 'b': latency_moderate_b, 'c': latency_moderate_c}},
                {'function': 'triangle_mf', 'term': 'high', 'params': {'a': latency_high_a, 'b': latency_high_b, 'c': latency_high_c}},
            ],
        },
        'OutBandwidth': {
            'universe_of_discourse': [0, 1],
            'linguistic_variables': [
                {'function': 'triangle_mf', 'term': 'low', 'params': {'a': out_bandwidth_low_a, 'b': out_bandwidth_low_b, 'c': out_bandwidth_low_c}},
                {'function': 'triangle_mf', 'term': 'medium', 'params': {'a': out_bandwidth_medium_a, 'b': out_bandwidth_medium_b, 'c': out_bandwidth_medium_c}},
                {'function': 'trapezoid_mf', 'term': 'high', 'params': {'a': out_bandwidth_high_a, 'b': out_bandwidth_high_b, 'c': out_bandwidth_high_c, 'd': out_bandwidth_high_c}},
            ],
        },
        'CLP': {
            'universe_of_discourse': [-1, 1],
            'linguistic_variables': [
                {'function': 'triangle_mf', 'term': 'decrease_significantly', 'params': {'a': clp_decrease_significantly_a, 'b': clp_decrease_significantly_b, 'c': clp_decrease_significantly_c}},
                {'function': 'triangle_mf', 'term': 'decrease', 'params': {'a': clp_decrease_a, 'b': clp_decrease_b, 'c': clp_decrease_c}},
                {'function': 'triangle_mf', 'term': 'maintain', 'params': {'a': clp_maintain_a, 'b': clp_maintain_b, 'c': clp_maintain_c}},
                {'function': 'triangle_mf', 'term': 'increase', 'params': {'a': clp_increase_a, 'b': clp_increase_b, 'c': clp_increase_c}},
                {'function': 'triangle_mf', 'term': 'increase_significantly', 'params': {'a': clp_increase_significantly_a, 'b': clp_increase_significantly_b, 'c': clp_increase_significantly_c}},
            ],
        },
    }

def sample_bellv9_hparams(trial: optuna.Trial):
    system_load_very_low_a_bell = trial.suggest_float("system_load_very_low_a_bell", 1, 4)
    system_load_very_low_b_bell = trial.suggest_float("system_load_very_low_b_bell", 0, 1)
    system_load_very_low_c_bell = trial.suggest_float("system_load_very_low_c_bell", 0, 0.3)

    system_load_low_a = trial.suggest_float("system_load_low_a", 1, 4)
    system_load_low_b = trial.suggest_float("system_load_low_b", 0, 1)
    system_load_low_c = trial.suggest_float("system_load_low_c", 0.2, 0.5)

    system_load_medium_a = trial.suggest_float("system_load_medium_a", 1, 4)
    system_load_medium_b = trial.suggest_float("system_load_medium_b", 0, 1)
    system_load_medium_c = trial.suggest_float("system_load_medium_c", 0.4, 0.7)

    system_load_high_a = trial.suggest_float("system_load_high_a", 1, 4)
    system_load_high_b = trial.suggest_float("system_load_high_b", 0, 1)
    system_load_high_c = trial.suggest_float("system_load_high_c", 0.6, 0.9)

    system_load_very_high_a_tri = trial.suggest_float("system_load_very_high_a_tri", 0.6, 1)

    # Latency Hyperparameters
    latency_low_a = trial.suggest_float("latency_low_a", 1, 4)
    latency_low_b = trial.suggest_float("latency_low_b", 0, 1)
    latency_low_c = trial.suggest_float("latency_low_c", 0, 0.4)

    latency_medium_a = trial.suggest_float("latency_medium_a", 1, 4)
    latency_medium_b = trial.suggest_float("latency_medium_b", 0, 1)
    latency_medium_c = trial.suggest_float("latency_medium_c", 0.2, 0.7)

    latency_high_a = trial.suggest_float("latency_high_a", 1, 4)
    latency_high_b = trial.suggest_float("latency_high_b", 0, 1)
    latency_high_c = trial.suggest_float("latency_high_c", 0.7, 1)

    # CLP Hyperparameters
    clp_very_low_c_tri = trial.suggest_float("clp_very_low_c_tri", -0.8, -0.6)

    clp_low_a = trial.suggest_float("clp_low_a", 1, 4)
    clp_low_b = trial.suggest_float("clp_low_b", 0, 1)
    clp_low_c = trial.suggest_float("clp_low_c", -0.7, -0.1)

    clp_medium_a = trial.suggest_float("clp_medium_a", 1, 4)
    clp_medium_b = trial.suggest_float("clp_medium_b", 0, 1)
    clp_medium_c = trial.suggest_float("clp_medium_c", -0.3, 0.6)

    clp_high_a = trial.suggest_float("clp_high_a", 1, 4)
    clp_high_b = trial.suggest_float("clp_high_b", 0, 1)
    clp_high_c = trial.suggest_float("clp_high_c", 0.3, 0.8)

    clp_very_high_a_tri = trial.suggest_float("clp_very_high_a_tri", 0.6, 1)

    return {
        'SystemLoad': {
            'universe_of_discourse': [0, 1],
            'linguistic_variables': [
                {
                    'function': 'bell_mf',
                    'term': 'very_low',
                    'params': {'a': system_load_very_low_a_bell, 'b': system_load_very_low_b_bell, 'c': system_load_very_low_c_bell},
                },
                {
                    'function': 'bell_mf',
                    'term': 'low',
                    'params': {'a': system_load_low_a, 'b': system_load_low_b, 'c': system_load_low_c},
                },
                {
                    'function': 'bell_mf',
                    'term': 'medium',
                    'params': {'a': system_load_medium_a, 'b': system_load_medium_b, 'c': system_load_medium_c},
                },
                {
                    'function': 'bell_mf',
                    'term': 'high',
                    'params': {'a': system_load_high_a, 'b': system_load_high_b, 'c': system_load_high_c},
                },
                {
                    'function': 'triangle_mf',
                    'term': 'very_high',
                    'params': {'a': 0.8, 'b': 1, 'c': 1},
                },
                {
                    'function': 'triangle_mf',
                    'term': 'very_high',
                    'params': {'a': system_load_very_high_a_tri, 'b': 1, 'c': 1},
                },
            ],
        },
        'Latency': {
            'universe_of_discourse': [0, 1],
            'linguistic_variables': [
                {
                    'function': 'bell_mf',
                    'term': 'low',
                    'params': {'a': latency_low_a, 'b': latency_low_b, 'c': latency_low_c},
                },
                {
                    'function': 'bell_mf',
                    'term': 'medium',
                    'params': {'a': latency_medium_a, 'b': latency_medium_b, 'c': latency_medium_c},
                },
                {
                    'function': 'bell_mf',
                    'term': 'high',
                    'params': {'a': latency_high_a, 'b': latency_high_b, 'c': latency_high_c},
                },
            ],
        },
        'CLP': {
            'universe_of_discourse': [-1, 1],
            'linguistic_variables': [
                {
                    'function': 'triangle_mf',
                    'term': 'decrease_significantly',
                    'params': {'a': -1, 'b': -1, 'c': clp_very_low_c_tri},
                },
                {
                    'function': 'bell_mf',
                    'term': 'decrease',
                    'params': {'a': clp_low_a, 'b': clp_low_b, 'c': clp_low_c},
                },
                {
                    'function': 'bell_mf',
                    'term': 'maintain',
                    'params': {'a': clp_medium_a, 'b': clp_medium_b, 'c': clp_medium_c},
                },
                {
                    'function': 'bell_mf',
                    'term': 'increase',
                    'params': {'a': clp_high_a, 'b': clp_high_b, 'c': clp_high_c},
                },
                {
                    'function': 'triangle_mf',
                    'term': 'increase_significantly',
                    'params': {'a': clp_very_high_a_tri, 'b': 1, 'c': 1},
                },
            ],
        },
    }


def mean_squared_error(y, y_pred):
    return (((y-y_pred)**2)/len(y)).sum()


def objective(trial):
    """
    The objective function for hyperparameter tuning
    :param trial: Used for defining the search space of the hyperparameters
    :return: the score
    """
    # Load dataset
    df = pd.read_csv('../input/CINTE24-25_Proj1_SampleData.csv')


    # Suggest hyperparameters
    hparams = sample_bellv9_hparams(trial)
    #hparams = sample_triangular_hparams(trial)
    # Train and evaluate model
    model = create_fuzzy_system(hparams)
    y_pred = model.predict(df)
    score = mean_squared_error(df['CLPVariation'], y_pred)
    return score


# Create a study object
study = optuna.create_study(direction="minimize")
study.enqueue_trial(mamdani_bellv9_best_params)  # start off with the previously best result
#study.enqueue_trial(mamdani_triangle_best_params)  # start off with the previously best result

try:
    # Optimize the objective function
    study.optimize(objective, n_trials=1_000, timeout=60)
except KeyboardInterrupt:
    print("Stopping optimization")

print("Number of finished trials: ", len(study.trials))
print("Best hyperparameters:", study.best_params)
print("FS hparams: ", convert_optuna_to_hparams(study.best_params))
#try:
#    print("FS hparams: ", convert_optuna_to_hparams_tri(study.best_params))
#except:
#    pass
print("Best value:", study.best_value)
fig1 = vis.plot_optimization_history(study)
fig2 = vis.plot_param_importances(study)
fig3 = vis.plot_slice(study)
fig1.show()
fig2.show()
fig3.show()
