from simpful import FuzzySystem, TriangleFuzzySet, AutoTriangle, LinguisticVariable, FuzzySet, Gaussian_MF
from fuzzy.fuzzy_system_wrapper import FuzzySystemWrapper


FS = FuzzySystemWrapper()


# Memory Usage Avg [%]
M1 = FuzzySet(function=Gaussian_MF(mu=0,sigma=0.40), term="low")  # TODO add an overlap between Fuzzy sets
M2 = FuzzySet(function=Gaussian_MF(mu=50,sigma=0.20), term="medium")
M3 = FuzzySet(function=Gaussian_MF(mu=0.85,sigma=0.20), term="high")
M4 = FuzzySet(function=Gaussian_MF(mu=1,sigma=0.15), term="critical")
FS.add_linguistic_variable("MemoryUsage", LinguisticVariable([M1,M2,M3,M4], universe_of_discourse=[0,1]))


# Processor Load [%]
P1 = FuzzySet(function=Gaussian_MF(mu=0,sigma=0.30), term="low")  # TODO add an overlap between Fuzzy sets
P2 = FuzzySet(function=Gaussian_MF(mu=0.50,sigma=0.15), term="medium")
P3 = FuzzySet(function=Gaussian_MF(mu=0.75,sigma=0.15), term="high")
P4 = FuzzySet(function=Gaussian_MF(mu=0.95,sigma=0.15), term="critical")
FS.add_linguistic_variable("ProcessorLoad", LinguisticVariable([P1,P2,P3,P4], universe_of_discourse=[0,1]))

# Latency [mS]
# from https://www.centurylink.com/home/help/internet/how-to-improve-gaming-latency.html

L1 = FuzzySet(function=Gaussian_MF(mu=0,sigma=30), term="great")  # TODO add an overlap between Fuzzy sets
L2 = FuzzySet(function=Gaussian_MF(mu=35,sigma=20), term="good")
L3 = FuzzySet(function=Gaussian_MF(mu=70,sigma=30), term="fair")
L4 = FuzzySet(function=Gaussian_MF(mu=200,sigma=100), term="poor")
#P4 = TriangleFuzzySet(280,500,500, term = "awful")
FS.add_linguistic_variable("Latency", LinguisticVariable([L1,L2,L3, L4], universe_of_discourse=[0,300]))

# CLP Variation (output)
CLP3 = FuzzySet(function=Gaussian_MF(mu=-1,sigma=0.2), term="decrease")  # TODO add an overlap between Fuzzy sets
CLP2 = FuzzySet(function=Gaussian_MF(mu=0,sigma=0.2), term="maintain")
CLP1 = FuzzySet(function=Gaussian_MF(mu=1,sigma=0.2), term="increase")
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

    save_path= '../output/mamdani_gaussian'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plot_inputs_outputs_fuzzy_system(FS, save_path)
    plot_memory_processor_clp(FS, save_path)

    with open(f"{save_path}/model.pkl", "wb") as f:
        pickle.dump(FS, f)




