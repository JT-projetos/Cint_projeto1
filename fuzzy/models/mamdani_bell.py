from simpful import FuzzySystem, LinguisticVariable, FuzzySet
from fuzzy.models.bell_mf import Bell_MF

FS = FuzzySystem()


# Memory Usage Avg [%]
M1 = FuzzySet(function=Bell_MF(a=2, b=0.25, c=0), term="low")
M2 = FuzzySet(function=Bell_MF(a=2, b=0.25, c=0.5), term="medium")
M3 = FuzzySet(function=Bell_MF(a=2, b=0.25, c=0.75), term="high")
M4 = FuzzySet(function=Bell_MF(a=2, b=0.25, c=1), term="critical")
FS.add_linguistic_variable("MemoryUsage", LinguisticVariable([M1,M2,M3,M4], universe_of_discourse=[0,1]))


# Processor Load [%]
P1 = FuzzySet(function=Bell_MF(a=2, b=0.25, c=0), term="low")
P2 = FuzzySet(function=Bell_MF(a=2, b=0.25, c=0.5), term="medium")
P3 = FuzzySet(function=Bell_MF(a=2, b=0.25, c=0.75), term="high")
P4 = FuzzySet(function=Bell_MF(a=2, b=0.25, c=1), term="critical")
FS.add_linguistic_variable("ProcessorLoad", LinguisticVariable([P1,P2,P3,P4], universe_of_discourse=[0,1]))

# CLP Variation (output)
CLP3 = FuzzySet(function=Bell_MF(a=2, b=0.3, c=-1), term="decrease")
CLP2 = FuzzySet(function=Bell_MF(a=2, b=0.3, c=0), term="maintain")
CLP1 = FuzzySet(function=Bell_MF(a=2, b=0.3, c=1), term="increase")
FS.add_linguistic_variable("CLP", LinguisticVariable([CLP1, CLP2, CLP3], universe_of_discourse=[-1,1]))

FS.add_rules([
    #"IF (Latency IS poor) THEN (CLP IS increase)",
    #"IF (MemoryUsage IS medium) AND (ProcessorLoad IS medium) THEN (CLP IS increase)",
    #"IF (MemoryUsage IS low) AND (ProcessorLoad IS medium) THEN (CLP IS increase)",
    #"IF (MemoryUsage IS medium) AND (ProcessorLoad IS low) THEN (CLP IS increase)",
    "IF (MemoryUsage IS low) AND (ProcessorLoad IS low) THEN (CLP IS increase)",
    "IF (MemoryUsage IS critical) OR (ProcessorLoad IS critical) THEN (CLP IS decrease)",
    #"IF (MemoryUsage IS high) OR (ProcessorLoad IS high) THEN (CLP IS maintain)",
])

if __name__ == '__main__':
    import os
    import pickle
    from fuzzy.visualization import *

    save_path = '../output/mamdani_bell'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plot_inputs_outputs_fuzzy_system(FS, save_path)
    #plot_memory_processor_clp(FS, save_path)

    with open(f"{save_path}/model.pkl", "wb") as f:
        pickle.dump(FS, f)
