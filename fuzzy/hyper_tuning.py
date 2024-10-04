import optuna
# from optuna import trial
import optuna.visualization as vis
import pandas as pd
from fuzzy.models.mamdani_hparams import create_fuzzy_system


def sample_hparams(trial: optuna.Trial):
    # System Load Hyperparameters
    #system_load_very_low_fn = trial.suggest_categorical("system_load_very_low_fn", ['bell_mf', 'triangle_mf'])
    #system_load_very_high_fn = trial.suggest_categorical("system_load_very_high_fn", ['bell_mf', 'triangle_mf'])

    #
    # system_load_very_low_fn = 'bell_mf'
    # system_load_very_high_fn = 'triangle_mf'
    #
    # match system_load_very_low_fn:
    #     case 'bell_mf':
    #         system_load_very_low_a_bell = trial.suggest_float("system_load_very_low_a_bell", 1, 4)
    #         system_load_very_low_b_bell = trial.suggest_float("system_load_very_low_b_bell", 0, 1)
    #         system_load_very_low_c_bell = trial.suggest_float("system_load_very_low_c_bell", 0, 0.3)
    #         #system_load_very_low_params = {'a': system_load_very_low_a_bell, 'b': system_load_very_low_b_bell, 'c': system_load_very_low_c_bell}
    #     # case 'triangle_mf':
    #     #     system_load_very_low_c = trial.suggest_float("system_load_very_low_c_tri", 0, 0.3)
    #     #     system_load_very_low_params = {'a': 0, 'b': 0, 'c': system_load_very_low_c}
    #     case _: raise RuntimeError("Non valid option selected for system_load_very_low_fn")

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

    # match system_load_very_high_fn:
    #     # case 'bell_mf':
    #     #     system_load_very_high_a = trial.suggest_float("system_load_very_high_a_bell", 1, 4)
    #     #     system_load_very_high_b = trial.suggest_float("system_load_very_high_b_bell", 0, 1)
    #     #     system_load_very_high_c = trial.suggest_float("system_load_very_high_c_bell", 0.7, 1)
    #     #     system_load_very_high_params = {'a': system_load_very_high_a, 'b': system_load_very_high_b,
    #     #                                    'c': system_load_very_high_c}
    #     case 'triangle_mf':
    #         system_load_very_high_a_tri = trial.suggest_float("system_load_very_high_a_tri", 0.6, 1)
    #         #system_load_very_high_params = {'a': system_load_very_high_a_tri, 'b': 1, 'c': 1}
    #     case _ as e: raise RuntimeError("Non valid option selected for system_load_very_low_fn")

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
    #clp_very_low_fn = trial.suggest_categorical("clp_very_low_fn", ['bell_mf', 'triangle_mf'])
    #clp_very_high_fn = trial.suggest_categorical("clp_very_high_fn", ['bell_mf', 'triangle_mf'])
    # clp_very_low_fn = 'triangle_mf'
    # clp_very_high_fn = 'triangle_mf'
    # match clp_very_low_fn:
    #     # case 'bell_mf':
    #     #     clp_very_low_a = trial.suggest_float("clp_very_low_a_bell", 1, 4)
    #     #     clp_very_low_b = trial.suggest_float("clp_very_low_b_bell", 0, 1)
    #     #     clp_very_low_c = trial.suggest_float("clp_very_low_c_bell", -1, -0.5)
    #     #     clp_very_low_params = {'a': clp_very_low_a, 'b': clp_very_low_b,
    #     #                                    'c': clp_very_low_c}
    #     case 'triangle_mf':
    #         clp_very_low_c_tri = trial.suggest_float("clp_very_low_c_tri", -0.8, -0.6)
    #         #clp_very_low_params = {'a': -1, 'b': -1, 'c': clp_very_low_c_tri}
    #     case _:
    #         raise RuntimeError("Non valid option selected for clp_very_low_fn")

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
    # match clp_very_high_fn:
    #     # case 'bell_mf':
    #     #     clp_very_high_a = trial.suggest_float("clp_very_high_a_bell", 1, 4)
    #     #     clp_very_high_b = trial.suggest_float("clp_very_high_b_bell", 0, 1)
    #     #     clp_very_high_c = trial.suggest_float("clp_very_high_c_bell", 0.7, 1)
    #     #     clp_very_high_params = {'a': clp_very_high_a, 'b': clp_very_high_b,
    #     #                                     'c': clp_very_high_c}
    #     case 'triangle_mf':
    #         clp_very_high_a_tri = trial.suggest_float("clp_very_high_a_tri", 0.6, 1)
    #         #clp_very_high_params = {'a': clp_very_high_a_tri, 'b': 1, 'c': 1}
    #     case _ as e:
    #         raise RuntimeError("Non valid option selected for clp_very_high_fn")

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
    hparams = sample_hparams(trial)
    # Train and evaluate model
    model = create_fuzzy_system(hparams)
    y_pred = model.predict(df)
    score = mean_squared_error(df['CLPVariation'], y_pred)
    return score


# Create a study object
study = optuna.create_study(direction="minimize")
study.enqueue_trial({
    #'system_load_very_low_fn': 'bell_mf',
    #'system_load_very_high_fn': 'triangle_mf',
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
    #'clp_very_low_fn': 'triangle_mf',
    #'clp_very_high_fn': 'triangle_mf',
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
})

try:
    # Optimize the objective function
    study.optimize(objective, n_trials=1, timeout=600)
except KeyboardInterrupt:
    print("Stopping optimization")

print("Number of finished trials: ", len(study.trials))
print("Best hyperparameters:", study.best_params)
print("Best value:", study.best_value)
#fig1 = vis.plot_optimization_history(study)
#fig2 = vis.plot_param_importances(study)
#fig3 = vis.plot_slice(study)
#fig1.show()
#fig2.show()
#fig3.show()

#
# trial = study.best_trial
# print("  Params: ")
# for key, value in trial.params.items():
#     print("    {}: {}".format(key, value))
