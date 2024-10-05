from simpful import FuzzySystem
import pandas as pd
import numpy as np


class FuzzySystemWrapper(FuzzySystem):
    def _set_row(self, row):
        self.set_variable('MemoryUsage', row['MemoryUsage'])
        self.set_variable('ProcessorLoad', row['ProcessorLoad'])
        self.set_variable('InpNetThroughput', row['InpNetThroughput'])
        self.set_variable('OutNetThroughput', row['OutNetThroughput'])
        self.set_variable('OutBandwidth', row['OutBandwidth'])
        self.set_variable('Latency', row['Latency'])
        self.set_variable('V_MemoryUsage', row['V_MemoryUsage'])
        self.set_variable('V_ProcessorLoad', row['V_ProcessorLoad'])
        self.set_variable('V_InpNetThroughput', row['V_InpNetThroughput'])
        self.set_variable('V_OutNetThroughput', row['V_OutNetThroughput'])
        self.set_variable('V_OutBandwidth', row['V_OutBandwidth'])
        self.set_variable('V_Latency', row['V_Latency'])

        SystemLoad = max(row['MemoryUsage'], row['ProcessorLoad'])
        self.set_variable("SystemLoad", SystemLoad)

    def predict(self, df: pd.DataFrame, show_bar=False) -> np.array:
        fs_results = []
        if not show_bar:
            for i, row in df.iterrows():
                self._set_row(row)
                fs_results.append(self.inference()['CLP'])
        else:
            from tqdm import tqdm

            for i, row in tqdm(df.iterrows(), total=df.shape[0]):
                self._set_row(row)
                fs_results.append(self.inference()['CLP'])
        return np.array(fs_results)
