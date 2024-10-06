from fuzzy.models.mamdani_triangle_v4 import FS
import numpy as np
import pandas as pd



sample_distribution = np.random.uniform

N_SAMPLES = 10_000
df = pd.DataFrame({
    'MemoryUsage': sample_distribution(size=N_SAMPLES),
    'ProcessorLoad': sample_distribution(size=N_SAMPLES),
    'InpNetThroughput': sample_distribution(size=N_SAMPLES),
    'OutNetThroughput': sample_distribution(size=N_SAMPLES),
    'OutBandwidth': sample_distribution(size=N_SAMPLES),
    'Latency': sample_distribution(size=N_SAMPLES),
    'V_MemoryUsage': sample_distribution(size=N_SAMPLES),
    'V_ProcessorLoad': sample_distribution(size=N_SAMPLES),
    'V_InpNetThroughput': sample_distribution(size=N_SAMPLES),
    'V_OutNetThroughput': sample_distribution(size=N_SAMPLES),
    'V_OutBandwidth': sample_distribution(size=N_SAMPLES),
    'V_Latency': sample_distribution(size=N_SAMPLES),
    #'CLPVariation': sample_distribution(low=-1, high=1, size=N_SAMPLES),
})

clp = FS.predict(df, show_bar=True)
df['CLPVariation'] = clp
print(df[['MemoryUsage', 'ProcessorLoad', 'Latency', 'CLPVariation']].head())
df.to_csv(f'../gen_input/{sample_distribution.__name__}{N_SAMPLES}.csv', index=False)
