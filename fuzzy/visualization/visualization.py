import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.graph_objs as go
from tqdm import tqdm
from fuzzy.visualization.fuzzy_system_to_dataframe import fuzzy_system_to_dataframe

#from main import FS
import os


def plot_inputs_outputs_fuzzy_system(FS, save_path=None):
    df = fuzzy_system_to_dataframe(FS)

    lvs = df['linguistic_var'].unique()
    for lv in lvs:
        ax = sns.lineplot(df[df['linguistic_var'] == lv], x='x', y='y', hue='term')
        ax.set_title(f"Linguistic Variable {lv}")

        if save_path is not None:
            if not os.path.exists(save_path + '/io_graphs/'):
                os.mkdir(save_path + '/io_graphs/')

            plt.savefig(f'{save_path}/io_graphs/{lv}.png', bbox_inches='tight')

        plt.show()


# 3D Plot
def plot_memory_processor_surface(FS, save_path=None):
    # Create a grid of values for Temperature and Frequency
    memory_values = np.linspace(0, 100, 50)
    processor_values = np.linspace(0, 100, 50)

    # Create meshgrid for plotting
    M, P = np.meshgrid(memory_values, processor_values)
    CLP = np.zeros_like(M)

    # Set Latency as a constant value
    FS.set_variable("Latency", 20)

    for i in tqdm(range(M.shape[0])):
        for j in range(M.shape[1]):
            FS.set_variable("MemoryUsage", M[i, j])
            FS.set_variable("ProcessorLoad", P[i, j])

            CLP[i, j] = FS.inference()["CLP"]

    fig = go.Figure(data=[go.Surface(x=P, y=M, z=CLP)])
    fig.update_layout(title='Fuzzy CLP Inference', autosize=False,
                      width=800, height=800,
                      margin=dict(l=65, r=50, b=65, t=90)
                      )
    fig.update_scenes(xaxis_title_text='memory [%]',
                      yaxis_title_text='processor [%]',
                      zaxis_title_text='CLP [%]')

    if save_path is not None:
        if not os.path.exists(save_path + '/memory_processor_3d'):
            os.mkdir(save_path + '/memory_processor_3d')

        fig.write_image(f"{save_path}/memory_processor_3d/memory_processor_3d.png")

        # also save results
        np.save(f"{save_path}/memory_processor_3d/clp.npy", CLP)

    fig.show()


if __name__ == '__main__':
    import pickle

    model_name = 'mamdani_triangular'

    # load fs
    with open(f'../output/{model_name}/model.pkl', 'rb') as f:
        FS = pickle.load(f)

    plot_inputs_outputs_fuzzy_system(FS)
    plot_memory_processor_surface(FS)
