from simpful import FuzzySystem, TriangleFuzzySet, AutoTriangle, LinguisticVariable
from fuzzy.fuzzy_system_wrapper import FuzzySystemWrapper


FS = FuzzySystemWrapper()

# System Load = Max (Memory Usage, Processor Load)
#S1 = TriangleFuzzySet(0,0,0.45, term = "low")
#S2 = TriangleFuzzySet(0.30,0.45,0.70, term = "moderate")
#S3 = TriangleFuzzySet(0.6,0.8,0.9, term = "high")
#S4 = TriangleFuzzySet(0.75,1,1, term = "critical")
#FS.add_linguistic_variable("SystemLoad", LinguisticVariable([S1,S2,S3,S4], universe_of_discourse=[0,1]))

S1 = TriangleFuzzySet(0,0,0.8, term = "low")
S2 = TriangleFuzzySet(0.4,0.45,0.5, term = "moderate")
S3 = TriangleFuzzySet(0.5,0.7,0.9, term = "high")
S4 = TriangleFuzzySet(0.65,1,1, term = "critical")
FS.add_linguistic_variable("SystemLoad", LinguisticVariable([S1, S2, S3, S4], universe_of_discourse=[0,1]))

# Latency
L1 = TriangleFuzzySet(a=0, b=0, c=0.5, term="low")
L2 = TriangleFuzzySet(a=0.4, b=0.6, c=0.8, term="medium")
L3 = TriangleFuzzySet(a=0.7, b=1, c=1, term="high")
FS.add_linguistic_variable("Latency", LinguisticVariable([L1, L2, L3], universe_of_discourse=[0,1]))


CLP1 = TriangleFuzzySet(0.6,1,1,   term="increase_significantly")
CLP2 = TriangleFuzzySet(-0.3,0.5,0.9,   term="increase")
#CLP3 = TriangleFuzzySet(-0.3,0,0.3,  term="maintain")
CLP4 = TriangleFuzzySet(-0.9,-0.5,0.3, term="decrease")
CLP5 = TriangleFuzzySet(-1,-1,-0.6, term="decrease_significantly")
FS.add_linguistic_variable("CLP", LinguisticVariable([CLP1, CLP2, CLP4, CLP5], universe_of_discourse=[-1,1]))

#FS.add_rules([
#    "IF (SystemLoad IS critical) THEN (CLP IS decrease_significantly)",
#    #"IF (SystemLoad IS high) AND (OutNetThroughput IS low) THEN (CLP IS mantain)",
#    #"IF (SystemLoad IS high) AND (OutNetThroughput IS moderate) THEN (CLP IS mantain)",
#    #"IF (SystemLoad IS high) AND (OutNetThroughput IS high) THEN (CLP IS decrease)",
#    #"IF (SystemLoad IS high) AND (OutNetThroughput IS very_high) THEN (CLP IS decrease)",
#    "IF (SystemLoad IS high) THEN (CLP IS increase)",
#    "IF (SystemLoad IS moderate) THEN (CLP IS increase)",
#    "IF (SystemLoad IS low) AND (OutNetThroughput IS low) THEN (CLP IS increase_significantly)",
#    "IF (SystemLoad IS low) AND (OutNetThroughput IS moderate) THEN (CLP IS increase_significantly)",
#    "IF (SystemLoad IS low) AND (OutNetThroughput IS high) THEN (CLP IS increase)",
#    "IF (SystemLoad IS low) AND (OutNetThroughput IS very_high) THEN (CLP IS increase)",
#])

FS.add_rules([

    "IF (SystemLoad IS critical) THEN (CLP IS decrease_significantly)",
    "IF (SystemLoad IS high) THEN (CLP IS increase_significantly)",
    "IF (SystemLoad IS low) THEN (CLP IS increase_significantly)",
    "IF (Latency IS high) AND (NOT (SystemLoad IS critical)) THEN (CLP IS increse_significantly)",
    #"IF (Latency IS high) AND ((SystemLoad IS very_low) OR (SystemLoad IS low) OR (SystemLoad IS medium)) THEN (CLP IS increse_significantly)",

])

if __name__ == '__main__':
    import os
    import pickle
    from fuzzy.visualization import *

    save_path= '../output/mamdani_triangular_v3'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plot_inputs_outputs_fuzzy_system(FS, save_path)
    plot_memory_processor_clp(FS, save_path)

    with open(f"{save_path}/model.pkl", "wb") as f:
        pickle.dump(FS, f)
