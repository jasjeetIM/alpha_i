import numpy as np

class NeuronType(object):
    phasic_spike=0
    tonic_spike=1
    phasic_burst=2
    tonic_burst=3
    inhib_spike=4
    inhib_burst=5
    resonate=6
    mixed_model=7 
    variable_thresh=8 
    accommodate=9
    rebound_spike=10 
    rebound_burst=11
    bistable=12

class Neuron(object):
    def __init__(self, neuronId, ntype,transmitter, columnId, layerId):
        self.nid = neuronId
        self.v = -70.0
        self.u = -20.0
        self.ntype = ntype
        #we don't create neurons that have no neighbors
        self.transmitter = transmitter
        self.layerId = layerId
        self.columnId = columnId
        

        
    def clearSpikeHistory(self, spikeLedger):
        if spikeLedger[self.nid]:
            spikeLedger[self.nid].clear()
    
    def updateWeights(self, spikeLedger, weightLedger, neighborLedger, lr_params, dynamics):
        self_size = len(spikeLedger[self.nid])
        if self_size == 0:
            self_spike = 0
        delta_w = 0.0
        #print(self.nid, self_size, len(neighborLedger[self.nid]))
        #Loop over every neighbor and update weights
        for i in range(len(neighborLedger[self.nid])):
            nei_size = len(spikeLedger[neighborLedger[self.nid][i]])
            if nei_size == 0:
                #print('nei size = 0')
                continue
            
            #Get first element from selfLedger if it exists, Else set it to 0
            self_ptr, nei_ptr = 0,0
            while self_ptr < self_size:
                self_spike = spikeLedger[self.nid][self_ptr]
                nei_spike = spikeLedger[neighborLedger[self.nid][i]][nei_ptr]
                if self_spike > nei_spike:
                    delta_w +=lr_params["A_pos"]/np.exp((self_spike - nei_spike)/lr_params["tau_pos"])
                    nei_ptr+=1
                    if nei_ptr >= nei_size:
                        break
                else:
                    self_ptr+=1
            while nei_ptr < nei_size:
                nei_spike = spikeLedger[neighborLedger[self.nid][i]][nei_ptr]
                delta_w += lr_params["A_neg"]/np.exp((nei_spike - self_spike)/lr_params["tau_neg"])
                nei_ptr+=1
            
            #Update the weight
            weightLedger[self.nid][i] = weightLedger[self.nid][i] + delta_w

        
    
    def updateLedger(self, spikeLedger, dynamics,time):
        spikeLedger[self.nid].append(time)
        #print(self.nid)
        self.v = dynamics[self.ntype]["c"]
        self.u = self.u + dynamics[self.ntype]["d"]
        
    
    def updateVoltage(self,spikeLedger,neighborLedger,weightLedger, time, dynamics, lr_params):
        #While calling updateVoltage at time T, the ledger will have spike history upto time T-1.
        I = dynamics[self.ntype]["I"]
        for i in range(len(neighborLedger[self.nid])):
            #Update I if neighbor fired
            if (spikeLedger[neighborLedger[self.nid][i]] and spikeLedger[neighborLedger[self.nid][i]][-1] == time-1):
                I = I + weightLedger[self.nid][i] # access weight for the ith neighbor

        #Update voltage and refactory period based on current voltage
        dv_dt = 0.04*self.v**2 + 5*self.v + 140 - self.u + I
        du_dt = dynamics[self.ntype]["a"]*(dynamics[self.ntype]["b"]*self.v - self.u)
        self.v = self.v +lr_params["alpha"]*dv_dt
        self.u = self.u +  lr_params["alpha"]*du_dt


