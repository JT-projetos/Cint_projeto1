from simpful import FuzzySystem, TriangleFuzzySet, AutoTriangle, LinguisticVariable
from fuzzy.fuzzy_system_wrapper import FuzzySystemWrapper


FS = FuzzySystemWrapper()

# Memory Usage Avg [%]
M1 = TriangleFuzzySet(0,0,0.4, term = "low")  # TODO add an overlap between Fuzzy sets
M2 = TriangleFuzzySet(0.3,0.5,0.70, term = "medium")
M3 = TriangleFuzzySet(0.60,0.85,1, term = "high")
M4 = TriangleFuzzySet(0.85,1,1, term = "critical")
FS.add_linguistic_variable("MemoryUsage", LinguisticVariable([M1,M2,M3,M4], universe_of_discourse=[0,1]))

# Processor Load [%]
P1 = TriangleFuzzySet(0,0,0.40, term = "low")
P2 = TriangleFuzzySet(0.30,0.50,0.70, term = "medium")
P3 = TriangleFuzzySet(0.60,0.85,1, term = "high")
P4 = TriangleFuzzySet(0.85,1,1, term = "critical")
FS.add_linguistic_variable("ProcessorLoad", LinguisticVariable([P1,P2,P3,P4], universe_of_discourse=[0,1]))

# Latency [mS]
# from https://www.centurylink.com/home/help/internet/how-to-improve-gaming-latency.html
L1 = TriangleFuzzySet(0,0,0.40, term = "great")
L2 = TriangleFuzzySet(0.30,0.50,0.70, term = "good")
L3 = TriangleFuzzySet(0.60,0.85,1, term = "fair")
L4 = TriangleFuzzySet(0.60,0.85,1, term = "poor")
#P4 = TriangleFuzzySet(280,500,500, term = "awful")
FS.add_linguistic_variable("Latency", LinguisticVariable([L1,L2,L3, L4], universe_of_discourse=[0,1]))

# CLP Variation (output)
CLP3 = TriangleFuzzySet(-1,-0.2,0, term="decrease")
CLP2 = TriangleFuzzySet(-0.7,0,0.7,  term="maintain")
CLP1 = TriangleFuzzySet(0,0.2,1,   term="increase")
FS.add_linguistic_variable("CLP", LinguisticVariable([CLP1, CLP2, CLP3], universe_of_discourse=[-1,1]))

FS.add_rules([
    "IF (Latency IS poor) THEN (CLP IS increase)",
    "IF (MemoryUsage IS medium) AND (ProcessorLoad IS medium) THEN (CLP IS increase)",
    "IF (MemoryUsage IS low) AND (ProcessorLoad IS medium) THEN (CLP IS increase)",
    "IF (MemoryUsage IS medium) AND (ProcessorLoad IS low) THEN (CLP IS increase)",
    "IF (MemoryUsage IS low) AND (ProcessorLoad IS low) THEN (CLP IS increase)",
    "IF (MemoryUsage IS critical) OR (ProcessorLoad IS critical) THEN (CLP IS decrease)",
    "IF (MemoryUsage IS high) OR (ProcessorLoad IS high) THEN (CLP IS maintain)"
])

if __name__ == '__main__':
    import os
    import pickle
    from fuzzy.visualization import *

    save_path= '../../output/mamdani_triangular'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plot_inputs_outputs_fuzzy_system(FS, save_path)
    plot_memory_processor_clp(FS, save_path)

    with open(f"{save_path}/model.pkl", "wb") as f:
        pickle.dump(FS, f)
