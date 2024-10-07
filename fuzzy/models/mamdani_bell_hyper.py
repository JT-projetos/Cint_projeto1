from fuzzy.models.mamdani_hparams import create_fuzzy_system


hparams = {"SystemLoad": {"universe_of_discourse": [0, 1], "linguistic_variables": [{"function": "bell_mf", "term": "very_low", "params": {"a": 2.3579646388364113, "b": 0.0677024659024898, "c": 0.10991925782543725}}, {"function": "bell_mf", "term": "low", "params": {"a": 2.9855565054404636, "b": 0.03761485406417333, "c": 0.4066707343866625}}, {"function": "bell_mf", "term": "medium", "params": {"a": 3.1909280088719183, "b": 0.07877078069866221, "c": 0.6217025436754964}}, {"function": "bell_mf", "term": "high", "params": {"a": 3.813887069664699, "b": 0.05347547026971319, "c": 0.8236234370371317}}, {"function": "triangle_mf", "term": "very_high", "params": {"a": 0.8, "b": 1, "c": 1}}, {"function": "triangle_mf", "term": "very_high", "params": {"a": 0.7703886470167463, "b": 1, "c": 1}}]}, "Latency": {"universe_of_discourse": [0, 1], "linguistic_variables": [{"function": "bell_mf", "term": "low", "params": {"a": 3.396390228114685, "b": 0.11095500510626258, "c": 0.20698390376560624}}, {"function": "bell_mf", "term": "medium", "params": {"a": 2.137127656649762, "b": 0.29557931953941596, "c": 0.38172707631883174}}, {"function": "bell_mf", "term": "high", "params": {"a": 2.814940737026227, "b": 0.21065806502605497, "c": 0.9367401553307901}}]}, "CLP": {"universe_of_discourse": [-1, 1], "linguistic_variables": [{"function": "triangle_mf", "term": "decrease_significantly", "params": {"a": -1, "b": -1, "c": -0.7355155336692256}}, {"function": "bell_mf", "term": "decrease", "params": {"a": 3.057673666369085, "b": 0.2629928283962844, "c": -0.44122058582907114}}, {"function": "bell_mf", "term": "maintain", "params": {"a": 2.7041119186309914, "b": 0.016997915653279842, "c": 0.08819860823068484}}, {"function": "bell_mf", "term": "increase", "params": {"a": 3.323870260594997, "b": 0.034564682743138384, "c": 0.7905031730917648}}, {"function": "triangle_mf", "term": "increase_significantly", "params": {"a": 0.6398899620938276, "b": 1, "c": 1}}]}}

FS = create_fuzzy_system(hparams=hparams)

if __name__ == '__main__':
    import os
    from fuzzy.visualization import *

    save_path = '../output/mamdani_bell_hyper'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plot_inputs_outputs_fuzzy_system(FS, save_path)

