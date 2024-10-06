from simpful import FuzzySystem, TriangleFuzzySet, AutoTriangle, LinguisticVariable, Trapezoidal_MF, FuzzySet
from fuzzy.fuzzy_system_wrapper import FuzzySystemWrapper


FS = FuzzySystemWrapper()


# System Load = Max (Memory Usage, Processor Load)
S1 = TriangleFuzzySet(0,0,0.5, term = "very_low")
S2 = TriangleFuzzySet(0.2,0.4,0.5, term = "low")
S3 = TriangleFuzzySet(0.4,0.5,0.6, term = "moderate")
S4 = TriangleFuzzySet(0.5,0.7,0.80, term = "high")
S5 = TriangleFuzzySet(0.7,1,1, term = "critical")
FS.add_linguistic_variable("SystemLoad", LinguisticVariable([S1,S2,S3,S4,S5], universe_of_discourse=[0,1]))

# latency
L1 = TriangleFuzzySet(0,0,0.8, term = "low")
L2 = TriangleFuzzySet(0.4,0.5,0.7, term = "moderate")
L3 = TriangleFuzzySet(0.4,1,1, term = "high")
#S2 = TriangleFuzzySet(0.30,0.45,0.70, term = "moderate")
#L4 = TriangleFuzzySet(0.65,1,1, term = "critical")
FS.add_linguistic_variable("Latency", LinguisticVariable([L1,L2,L3], universe_of_discourse=[0,1]))


# Output Bandwidth [bps]
B1 = TriangleFuzzySet(0,0,0.5, term = "low")
#T2 = TriangleFuzzySet(0.2,0.45,0.70, term = "moderate")
B3 = TriangleFuzzySet(0.3,0.5,0.8, term = "medium")
B4 = FuzzySet(function=Trapezoidal_MF(0.6,0.75,1,1), term = "high")
#B4 = TriangleFuzzySet(0.5,1,1, term = "high")
FS.add_linguistic_variable("OutBandwidth", LinguisticVariable([B1,B3, B4], universe_of_discourse=[0,1]))

# CLP Variation (output)
CLP1 = TriangleFuzzySet(0.6,1,1,   term="increase_significantly")
CLP2 = TriangleFuzzySet(0,0.5,0.9,   term="increase")
CLP3 = TriangleFuzzySet(-0.3,0,0.3,  term="maintain")
CLP4 = TriangleFuzzySet(-0.7,-0.3,-0.2, term="decrease")
CLP5 = TriangleFuzzySet(-1,-1,-0.6, term="decrease_significantly")
FS.add_linguistic_variable("CLP", LinguisticVariable([CLP1, CLP2, CLP3, CLP4, CLP5], universe_of_discourse=[-1,1]))


FS.add_rules([

    "IF (SystemLoad IS critical) AND (OutBandwidth IS high) THEN (CLP IS decrease_significantly)",

    "IF (SystemLoad IS critical) AND (OutBandwidth IS low) THEN (CLP IS decrease)",

    "IF (SystemLoad IS critical) AND (OutBandwidth IS medium) THEN (CLP IS decrease)",

    "IF (SystemLoad IS high) AND (Latency IS high) THEN (CLP IS increase_significantly)",

    "IF (SystemLoad IS high) AND (Latency IS low) THEN (CLP IS maintain)",

    "IF (SystemLoad IS moderate) THEN (CLP IS increase)",

    "IF (SystemLoad IS low) THEN (CLP IS increase_significantly)",

    "IF (SystemLoad IS very_low) THEN (CLP IS increase_significantly)",

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
