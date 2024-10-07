import pandas as pd


def classify(num) -> str:
    """Classify as {'Decrease', 'Increase', 'Maintain'} based on NN output (CLP)"""
    if -1 <= num <= -0.25:
        return 'Decrease'
    elif -0.25 < num <= 0.25:
        return 'Maintain'
    elif 0.25 < num <= 1:
        return 'Increase'
    else:
        raise ValueError("CLPVariation should be [-1, 1]")


if __name__ == '__main__':
    # input_file = '../../gen_input/bell_hyper_uniform100000.csv'
    # output_file = '../../gen_input/bell_hyper_uniform100000_class.csv'
    input_file = '../../gen_input/trianglev4_uniform10000.csv'
    output_file = '../../gen_input/trianglev4_uniform10000_class.csv'

    df = pd.read_csv(input_file, nrows=100000)

    df['fs_label'] = df['CLPVariation'].apply(classify)
    print(df[['CLPVariation', 'fs_label']].head())
    df.to_csv(output_file, index=False)
    #for i, row in df.iterrows():
