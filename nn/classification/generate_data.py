import pandas as pd


def classify(num) -> str:
    if -1 <= num <= 0.3:
        return 'Decrease'
    elif 0.3 < num <= 0.5:
        return 'Maintain'
    elif 0.5 < num <= 1:
        return 'Increase'
    else:
        raise ValueError("CLPVariation should be [-1, 1]")


input_file = '../../gen_input/uniform100000.csv'
output_file = '../../gen_input/uniform100000_class.csv'

df = pd.read_csv(input_file, nrows=100)

df['fs_label'] = df['CLPVariation'].apply(classify)
print(df[['CLPVariation', 'fs_label']].head())
df.to_csv(output_file, index=False)
#for i, row in df.iterrows():
