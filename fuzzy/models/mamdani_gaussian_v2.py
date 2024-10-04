from simpful import FuzzySystem, TriangleFuzzySet, AutoTriangle, LinguisticVariable, FuzzySet, Gaussian_MF
from fuzzy.fuzzy_system_wrapper import FuzzySystemWrapper


FS = FuzzySystemWrapper()

# System Load = Max (Memory Usage, Processor Load)
S1 = FuzzySet(function=Gaussian_MF(mu=0,sigma=0.133), term="low")  # TODO add an overlap between Fuzzy sets
S2 = FuzzySet(function=Gaussian_MF(mu=0.5,sigma=0.06), term="moderate")
S3 = FuzzySet(function=Gaussian_MF(mu=0.7,sigma=0.06), term="high")
S4 = FuzzySet(function=Gaussian_MF(mu=0.85,sigma=0.1), term="critical")
#P4 = TriangleFuzzySet(280,500,500, term = "awful")
FS.add_linguistic_variable("SystemLoad", LinguisticVariable([S1,S2,S3,S4], universe_of_discourse=[0,1]))

# Output Throughput [bps]
L1 = FuzzySet(function=Gaussian_MF(mu=0,sigma=0.1), term="low")  # TODO add an overlap between Fuzzy sets
L2 = FuzzySet(function=Gaussian_MF(mu=0.5,sigma=0.06), term="moderate")
L3 = FuzzySet(function=Gaussian_MF(mu=0.7,sigma=0.06), term="high")
L4 = FuzzySet(function=Gaussian_MF(mu=0.8,sigma=0.1), term="very_high")
#P4 = TriangleFuzzySet(280,500,500, term = "awful")
FS.add_linguistic_variable("Througthput", LinguisticVariable([L1,L2,L3, L4], universe_of_discourse=[0,1]))

# CLP Variation (output)
CLP1 = FuzzySet(function=Gaussian_MF(mu=0.8,sigma=0.06), term="increase significantly")  # TODO add an overlap between Fuzzy sets
CLP2 = FuzzySet(function=Gaussian_MF(mu=0.3,sigma=0.166), term="increase")
CLP3 = FuzzySet(function=Gaussian_MF(mu=0.0,sigma=0.1), term="maintain")
CLP4 = FuzzySet(function=Gaussian_MF(mu=-0.3,sigma=0.166), term="decrease")
CLP5 = FuzzySet(function=Gaussian_MF(mu=-0.8,sigma=0.06), term="decrease significantly")
FS.add_linguistic_variable("CLP", LinguisticVariable([CLP1, CLP2, CLP3, CLP4, CLP5], universe_of_discourse=[-1,1]))

FS.add_rules([
    "IF (SystemLoad IS critical) THEN (CLP IS decrease significantly)",
    "IF (SystemLoad IS high) AND (Througthput IS low) THEN (CLP IS mantain)",
    "IF (SystemLoad IS high) AND (Througthput IS moderate) THEN (CLP IS mantain)",
    "IF (SystemLoad IS high) AND (Througthput IS high) THEN (CLP IS decrease)",
    "IF (SystemLoad IS high) AND (Througthput IS very_high) THEN (CLP IS decrease)",
    "IF (SystemLoad IS moderate) THEN (CLP IS increase)",
    "IF (SystemLoad IS low) AND (Througthput IS low) THEN (CLP IS increase significantly)",
    "IF (SystemLoad IS low) AND (Througthput IS moderate) THEN (CLP IS increase significantly)",
    "IF (SystemLoad IS low) AND (Througthput IS high) THEN (CLP IS increase)",
    "IF (SystemLoad IS low) AND (Througthput IS very_high) THEN (CLP IS increase)",
])

if __name__ == '__main__':
    import os
    import pickle
    from fuzzy.visualization import *

    save_path= '../output/mamdani_gaussian_v2'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plot_inputs_outputs_fuzzy_system(FS, save_path)
    #plot_memory_processor_clp(FS, save_path)

    with open(f"{save_path}/model.pkl", "wb") as f:
        pickle.dump(FS, f)
