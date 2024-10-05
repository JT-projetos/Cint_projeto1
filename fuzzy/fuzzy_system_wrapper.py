from simpful import FuzzySystem
import pandas as pd
import numpy as np


class FuzzySystemWrapper(FuzzySystem):
    def predict(self, df: pd.DataFrame) -> np.array:
        fs_results = []
        # Maybe in future create a predict function common to all FS that receives df
        for i, row in df.iterrows():
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
            fs_results.append(self.inference()['CLP'])
        return np.array(fs_results)
