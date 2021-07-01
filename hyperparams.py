from neuron import NeuronType

neuronType = NeuronType
#Map with format: { a, b, c, d, I, alpha}
lr_params = {
    "A_pos": 0.1,
    "A_neg": 0.2,
    "tau_pos": 1.1,
    "tau_neg": 1.1,
    "alpha": 10**-2,
    "beta": 10**-6,
    "tau": 1000.
}
dynamics = {
    neuronType.phasic_spike : {"a": 0.02, "b": 0.25, "c": -65, "d": 6, "I": 0.5},
    neuronType.tonic_spike : {"a":  0.02,"b": 0.2, "c":-65, "d": 6, "I": 14},
    neuronType.phasic_burst :{"a": 0.02,"b": 0.25,"c":-55, "d":0.05, "I": 0.6},
    neuronType.tonic_burst : {"a": 0.02, "b":0.2,"c":-50, "d":2,"I":  15},
    neuronType.inhib_spike: {"a": -0.02, "b":-1,"c": -60, "d":8, "I": 80},
    neuronType.inhib_burst: {"a": -0.026,"b":-1,"c": -45, "d":0, "I": 80},
    neuronType.resonate : {"a": 0.1,"b": 0.26,"c":-60, "d":-1, "I": 0},
    neuronType.mixed_model : {"a": 0.02, "b":0.2,"c":-55, "d":4,"I": 10},
    neuronType.variable_thresh: {"a": 0.03, "b":0.25,"c":-60, "d":4, "I": 0},
    neuronType.accommodate: {"a": 0.02,"b":1,"c": -55, "d":4, "I": 0},
    neuronType.rebound_spike : {"a": 0.03,"b": 0.25,"c":-60, "d":4, "I": 0},
    neuronType.rebound_burst : {"a": 0.03, "b":0.25,"c":-52, "d":0, "I": 0},
    neuronType.bistable : {"a": 1., "b":1.5,"c":-60, "d":0,"I":  -65},
}
birth_alpha_by_ntype = {
    neuronType.phasic_spike:1,
    neuronType.tonic_spike:1,
    neuronType.phasic_burst:1e10,
    neuronType.tonic_burst:1e10,
    neuronType.inhib_spike:1,
    neuronType.inhib_burst:1,
    neuronType.resonate:1e10,
    neuronType.mixed_model:1,
    neuronType.variable_thresh:1,
    neuronType.accommodate:1,
    neuronType.rebound_spike:1,
    neuronType.rebound_burst:1,
    neuronType.bistable:1,
}


inhibitory_alpha = [20,70]
variance_param = 5.0 
mu_num_neurons_top = 7
mu_num_neurons_other = 75

sameCol = {}
parentCol = {}
childCol = {}
sibCol = {}
dist = {}

for i in range(6):
    for j in range(6):
        
        #Same column / Inter laminar
        if i == 0 and j == 1:
            sameCol[(i,j)] = 0.1
            parentCol[(i,j)] = 0.0
            childCol[(i,j)] = 0.0
            sibCol[(i,j)] = 0.0
            dist[(i,j)] = 0.0
        elif i == 3 and j == 2:
            sameCol[(i,j)] = 0.1
            parentCol[(i,j)] = 0.0
            childCol[(i,j)] = 0.0
            sibCol[(i,j)] = 0.0
            dist[(i,j)] = 0.0
        elif i == 3 and j == 1:
            sameCol[(i,j)] = 0.1
            parentCol[(i,j)] = 0.0
            childCol[(i,j)] = 0.0
            sibCol[(i,j)] = 0.0
            dist[(i,j)] = 0.0
        elif i == 2 and j == 1:
            sameCol[(i,j)] = 0.1
            parentCol[(i,j)] = 0.0
            childCol[(i,j)] = 0.0
            sibCol[(i,j)] = 0.0
            dist[(i,j)] = 0.0
        elif i == 3 and j == 4:
            sameCol[(i,j)] = 0.1
            parentCol[(i,j)] = 0.0
            childCol[(i,j)] = 0.0
            sibCol[(i,j)] = 0.0
            dist[(i,j)] = 0.0
        elif i == 4 and j == 5:
            sameCol[(i,j)] = 0.1
            parentCol[(i,j)] = 0.0
            childCol[(i,j)] = 0.0
            sibCol[(i,j)] = 0.0
            dist[(i,j)] = 0.0
        elif i == 0 and j == 4:
            sameCol[(i,j)] = 0.05
            parentCol[(i,j)] = 0.0
            childCol[(i,j)] = 0.0
            sibCol[(i,j)] = 0.0
            dist[(i,j)] = 0.0
        elif i == 2 and j == 3:
            sameCol[(i,j)] = 0.0
            parentCol[(i,j)] = 0.0
            childCol[(i,j)] = 0.05 #bottom up
            sibCol[(i,j)] = 0.0
            dist[(i,j)] = 0.05
        elif i == 1 and j == 3:
            sameCol[(i,j)] = 0.0
            parentCol[(i,j)] = 0.1
            childCol[(i,j)] = 0.1 #bottom up
            sibCol[(i,j)] = 0.05 #sibling
            dist[(i,j)] = 0.05
        elif i == 5 and j == 3:
            sameCol[(i,j)] = 0.0
            parentCol[(i,j)] = 0.0
            childCol[(i,j)] = 0.1 #bottom up
            sibCol[(i,j)] = 0.05 #sibling
            dist[(i,j)] = 0.05

            
        #top down inhibition
        elif i == 5 and j == 0:
            sameCol[(i,j)] = 0.0
            parentCol[(i,j)] = 0.01
            childCol[(i,j)] = 0.0
            dist[(i,j)] = 0.05
            sibCol[(i,j)] = 0.0
        elif i == 2 and j == 0:
            sameCol[(i,j)] = 0.0
            parentCol[(i,j)] = 0.1
            childCol[(i,j)] = 0.0
            dist[(i,j)] = 0.05
            sibCol[(i,j)] = 0.0

        else:
            sameCol[(i,j)] = 0.0
            parentCol[(i,j)] = 0.0
            childCol[(i,j)] = 0.0
            sibCol[(i,j)] = 0.0 #sibling
            dist[(i,j)] = 0.0

#Same Layer and same column
for i in range(6):
	sameCol[(i,i)] = 0.1


V1_numRoots = 5
V1_numLevels = 2
V1_offspring_prob = 0.5
V1_parent_prob = 0.5
V1_sibling_prob = 0.05
V1_mu_offspring= 3


# Relay station 
Thalmus_numRoots = 64
Thalmus_numLevels = 1
Thalmus_offspring_prob = 1.0
Thalmus_parent_prob = 0.5
Thalmus_sibling_prob = 0.1
Thalmus_mu_offspring= 4


#feed forward network
Motor_numRoots = 3
Motor_numLevels = 2
Motor_offspring_prob = 1.0 
Motor_parent_prob = 1.0
Motor_sibling_prob = 0.05
Motor_mu_offspring= 4



HC_numRoots = 10
HC_numLevels = 1
HC_offspring_prob = 0.0
HC_parent_prob = 0.0
HC_sibling_prob = 0.75
HC_mu_offspring= 4


outputlevels, inputlevels = {}, {}
outputlevels['TH_V1'] = 0
outputlevels['V1_TH'] = 0 
outputlevels['V1_HC'] = 0 
outputlevels['HC_V1'] = 0
outputlevels['TH_MT'] = 0
outputlevels['V1_MT'] = 0 
outputlevels['MT_TH'] = 0
outputlevels['TH_HC'] = 0
outputlevels['HC_TH'] = 0
inputlevels['TH_V1'] = V1_numLevels-1
inputlevels['V1_TH'] = 0
inputlevels['V1_HC'] = 0 
inputlevels['HC_V1'] = 0
inputlevels['TH_MT'] = 0
inputlevels['V1_MT'] = 0 
inputlevels['MT_TH'] = 0
inputlevels['TH_HC'] = 0
inputlevels['HC_TH'] = 0
