import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
from tqdm import tqdm
from fuzzy.visualization.fuzzy_system_to_dataframe import fuzzy_system_to_dataframe

#from main import FS
import os

sns.set_style('darkgrid')


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
def plot_memory_processor_clp(FS, save_path=None):
    # Create a grid of values for Temperature and Frequency
    load_values = np.linspace(0, 1, 50)
    Througthput_values = np.linspace(0, 1, 50)
    MemoryUsage = np.linspace(0, 1, 50)
    ProcessorLoad = np.linspace(0, 1, 50)

    # Create meshgrid for plotting
    #M, P = np.meshgrid(load_values, Througthput_values)
    #CLP = np.zeros_like(M)

    # Create meshgrid for plotting
    M, P = np.meshgrid(MemoryUsage, ProcessorLoad)
    CLP = np.zeros_like(M)

    #Set Latency as a constant value
    FS.set_variable("Latency", 0.5)

    for i in tqdm(range(M.shape[0])):
        for j in range(M.shape[1]):
            FS.set_variable("MemoryUsage", M[i, j])
            FS.set_variable("ProcessorLoad", P[i, j])

            CLP[i, j] = FS.inference()["CLP"]

    fig = go.Figure(data=[go.Surface(x=M, y=P, z=CLP)])
    fig.update_layout(title='Fuzzy CLP Inference', autosize=False,
                      width=800, height=600,
                      margin=dict(l=65, r=50, b=65, t=90)
                      )
    fig.update_scenes(xaxis_title_text='MemoryUsage [%]',
                      yaxis_title_text='ProcessorLoad [%]',
                      zaxis_title_text='CLP [%]')

    if save_path is not None:
        if not os.path.exists(save_path + '/memory_processor_3d'):
            os.mkdir(save_path + '/memory_processor_3d')

        fig.write_image(f"{save_path}/memory_processor_3d/memory_processor_3d.png")

        # also save results
        np.save(f"{save_path}/memory_processor_3d/clp.npy", CLP)

    fig.show()


def plot_sample_data(file_path: str, save_path=None, filter_list: list = None):
    df = pd.read_csv(file_path)
    fig = px.line(df)
    fig.show()

    if save_path is not None:
        if filter_list is None:
            sns.lineplot(df)
        else:
            sns.lineplot(df[filter_list])
        # remove / from path then remove file extension
        file_name = file_path.split('/')[-1].split('.')[0]
        plt.savefig(f'{save_path}/{file_name}.png', bbox_inches='tight')
        plt.show()


def plot_memory_processor_clp_sample_data(file_path: str, save_path=None):

    df = pd.read_csv(file_path)
    M = df['MemoryUsage'].values
    P = df['ProcessorLoad'].values
    CLP = df['CLPVariation'].values
    #M, P = np.meshgrid(memory_values, processor_values)
    #CLP = np.zeros_like(M)

    fig = px.scatter_3d(df, x='MemoryUsage', y='ProcessorLoad', z='CLPVariation')
    # fig = go.Figure(data=[go.Surface(x=P, y=M, z=CLP)])
    fig.update_layout(title='Fuzzy CLP Inference', autosize=False,
                      #width=800, height=600,
                      margin=dict(l=65, r=50, b=65, t=90)
                      )
    fig.update_scenes(xaxis_title_text='memory [%]',
                      yaxis_title_text='processor [%]',
                      zaxis_title_text='CLP [%]')

    if save_path is not None:
        fig.write_image(f"{save_path}/memory_processor_sctter_3d.png")

    fig.show()


def plot_model_scores(file_path: str):
    df = pd.read_csv(file_path)
    df['Datapoint'] = np.arange(1, 11)
    df = pd.melt(df, id_vars='Datapoint', var_name='Model', value_name='Relative Error')
    #print(df)
    ax = sns.barplot(data=df, x='Datapoint', y='Relative Error', hue='Model')
    ax.set_ylabel('Relative Error [%]')
    plt.show()

def plot_mse_scores(df):
    ax = sns.barplot(df)
    plt.show()


if __name__ == '__main__':
    import pickle

    model_name = 'mamdani_triangular'

    # load fs
    with open(f'../output/{model_name}/model.pkl', 'rb') as f:
        FS = pickle.load(f)

    #plot_inputs_outputs_fuzzy_system(FS)
    #plot_memory_processor_clp(FS)
    # plot_sample_data('../../input/CINTE24-25_Proj1_SampleData.csv', '../../input',
    #                  ['MemoryUsage', 'ProcessorLoad', 'Latency', 'CLPVariation'])
    #plot_memory_processor_clp_sample_data('../../input/CINTE24-25_Proj1_SampleData.csv', save_path='../../input/')
    plot_model_scores('../output/eval_models/model_scores.csv')
