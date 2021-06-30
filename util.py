import numpy as np
import cv2
from neuron import Neuron, NeuronType
import time

def resizeFrame(frame):
    frame = frame[30:-12,5:-4]
    frame = np.average(frame,axis = 2)
    frame = cv2.resize(frame,(84,84),interpolation = cv2.INTER_NEAREST)
    frame = np.array(frame,dtype = np.uint8)
    return frame

def getOutput(Motor_levelToColumns,columnToLayer, layerToNeuron ,spikes,trans,num_classes =3):
    output_columns_motor = Motor_levelToColumns[0]
    #account for inhibitory
    scores = np.zeros(num_classes)              
    for i,cols in enumerate(output_columns_motor):
        v = 0.0
        layer = columnToLayer[cols][5] #send actions via layer 6
        for neuron_id in layerToNeuron[layer]:
            v+=spikes[neuron_id]*trans[neuron_id]
        scores[i]=v

    final = np.argmax(scores)
    if final == 1:
        return 2
    elif final == 2:
        return 3
    else:
        return 0


def getGradientDataStruc(Motor_columns, columnToLayer,layerToNeuron, memory):
	neuron_count = 0
	idx_to_nid , nid_to_idx = {}, {}
	#account for inhibitory              
	for i,cols in enumerate(list(Motor_columns)):
		for l in range(6):
			layer = columnToLayer[cols][l] #send actions via layer 6
			for neuron_id in layerToNeuron[layer]:
				idx_to_nid[neuron_count] = neuron_id
				nid_to_idx[neuron_id] = neuron_count
				neuron_count+=1
	dv_dv = np.zeros((neuron_count,neuron_count,memory), dtype=np.float16)
	dv_dw = np.zeros((neuron_count,neuron_count,memory), dtype=np.float16)
	
	return dv_dv, dv_dw, idx_to_nid, nid_to_idx
	
	 
def gradientDescent(act, Motor_levelToColumns,columnToLayer, layerToNeuron,adj,idx_to_nid,nid_to_idx,spikes,trans, dv_dv, dv_dw, idx, window_size, alpha, a, b,neighbors, v):
	output_columns_motor = Motor_levelToColumns[0]
	total_loss = 0.
	curr_level = []
	#Get gradient wrt loss for target neurons              
	for i,cols in enumerate(output_columns_motor):
		if i == act:
			layer = columnToLayer[cols][5] #send actions via layer 6
			for neuron_id in layerToNeuron[layer]:
			#Neuron either didnt spike or spiked and was inhibitory
				if trans[neuron_id] < 0 and spikes[neuron_id, idx] == 1: 
					#Inhib should not have fired
					total_loss+=1					
					curr_level.append((neuron_id,1./100.))
				elif trans[neuron_id] > 0 and spikes[neuron_id,idx] == 0:
					total_loss+=1
					curr_level.append((neuron_id,-1./100.))

	print('Total Loss: {}'.format(total_loss))
	
	#BFS : gradient descent 
	total_delta = 0.
	curr_idx = idx - 1 
	d_curr_idx = window_size - 1 
	#This means that curr_idx represents derivative from the prev time step (idx - 1)
	# which we consider as the latest derivative (window_size -1)
	next_level = []
	mem = spikes.shape[1]
	total_steps = 0
	while curr_level and d_curr_idx >= 0:
		node, prev_ = curr_level.pop(0)
		neigh = adj[node]
		
		node_idx = nid_to_idx[node]
		#print(neigh)
		for nei in neigh:
			if nei not in adj:
				continue
			#print(nei)
			start = time.time()
			nei_idx = nid_to_idx[nei]
			#This will be true when we start GD
			if dv_dw[node_idx, nei_idx, d_curr_idx] == np.nan:
				#calculate dv_dw if we have 2 previous values
				if dv_dw[node_idx, nei_idx, d_curr_idx-1] != np.nan and dv_dw[node_idx, nei_idx, d_curr_idx-2] != np.nan:
					dw = 0.08*v[node,curr_idx-1]*dv_dw[node_idx, nei_idx, d_curr_idx-1] + \
						 5.*dv_dw[node_idx, nei_idx, d_curr_idx-1] - a*b*dv_dw[node_idx, nei_idx, d_curr_idx-2] + \
						((v[nei, curr_idx-1] + 70.0)/100.)*trans[nei]

				else:
					dw = ((v[nei, curr_idx-1] + 70.0)/100.)*trans[nei]

				dv_dw[node_idx, nei_idx, d_curr_idx] = dw
			
			#Update weight in a zig-zaggy manner. 
			# Weights will move up and down depending on derivative at various time steps
			neighbors[node, nei]= neighbors[node, nei] + alpha*prev_*dv_dw[node_idx, nei_idx, d_curr_idx]
			total_delta+=(alpha*prev_*dv_dw[node_idx, nei_idx, d_curr_idx])
			if dv_dv[node_idx, nei_idx, d_curr_idx] == np.nan:
				#calculate dv_dv
				if dv_dv[node_idx, nei_idx, d_curr_idx-1] != np.nan and dv_dv[node_idx, nei_idx, d_curr_idx-2] != np.nan:
					dv = 0.08*v[node,curr_idx-1]*dv_dv[node_idx, nei_idx, d_curr_idx-1] + \
						 5.*dv_dv[node_idx, nei_idx, d_curr_idx-1] - a*b*dv_dv[node_idx, nei_idx, d_curr_idx-2] + \
						neighbors[node, nei]*trans[nei]
				else:
					dv = neighbors[node, nei]*trans[nei]
				
				dv_dv[node_idx, nei_idx, d_curr_idx] = dv
			
			#Chain rule
			prev_ = prev_*dv_dv[node_idx, nei_idx, d_curr_idx]
			
			#BFS step
			next_level.append((nei, prev_))
			total_steps+=1
			print('Steps = {}, Time = {}'.format(total_steps, time.time() - start))
		
		if not curr_level:
			curr_level = next_level
			next_level = []
			curr_idx-=1
			d_curr_idx-=1
					
				
	dv_dw*=np.nan
	dv_dv*=np.nan
	print('GD: Total delta_w: {}'.format(total_delta))
	return neighbors
	
def stdp(neighbors, spikes, adj_motor, idx, memory, lr_params):
	total_delta=0
	for i in range(neighbors.shape[0]):
		if i not in adj_motor:
			delta_w = 0.0
			self_spikes = np.nonzero(spikes[i,:])[0]
			#Loop over every neighbor and update weights
			for nei in np.nonzero(neighbors[i])[0]:
				#print(nei)
				nei_spikes = np.nonzero(spikes[nei, :])[0]
				self_ptr, nei_ptr, self_size, nei_size = 0,0, len(self_spikes), len(nei_spikes)
				self_spike, nei_spike = 0,0
				if nei_size == 0:
					while self_ptr < self_size:
						self_spike = self_spikes[self_ptr]
						delta_w -= lr_params["A_neg"]/np.exp((self_spike - nei_spike)/lr_params["tau_neg"])
						self_ptr+=1
					neighbors[i, nei] = neighbors[i, nei] + delta_w
					continue

				while self_ptr < self_size:
					self_spike = self_spikes[self_ptr]
					nei_spike = nei_spikes[nei_ptr]
					if self_spike > nei_spike:
						delta_w +=lr_params["A_pos"]/np.exp((self_spike - nei_spike)/lr_params["tau_pos"])
						nei_ptr+=1
						if nei_ptr >= nei_size:
							break
					else:
						self_ptr+=1
			    
				while nei_ptr < nei_size:
					nei_spike = nei_spikes[nei_ptr]
					delta_w -= lr_params["A_neg"]/np.exp((nei_spike - self_spike)/lr_params["tau_neg"])
					nei_ptr+=1
				#Update the weight
				neighbors[i, nei] = neighbors[i, nei] + delta_w
				total_delta+=delta_w
				del nei_spikes
			del self_spikes

				
			
	print('Total delta_w: {}'.format(total_delta))
	return neighbors

def connectRegion(columns, parentEdges, siblingEdges, childEdges, columnToLayer,layerIdToNum,sameCol, parentCol,childCol, sibCol, layerToNeuron, neighbors, numConnections): 
    startConnections = numConnections
    for col1 in columns:
        for col2 in columns:
        #there is a connection between two columns
            if col1 == col2 or (col1,col2) in parentEdges or (col1,col2) in siblingEdges or (col1,col2) in childEdges:
                for layer_id_col1 in columnToLayer[col1]:
                    for layer_id_col2 in columnToLayer[col2]:
                        layer_col1 = layerIdToNum[layer_id_col1]
                        layer_col2 = layerIdToNum[layer_id_col2]
                        if col1 == col2:
                            conn_prob = sameCol[(layer_col1, layer_col2)]
                        #parent
                        elif (col1,col2) in parentEdges:
                            conn_prob = parentCol[(layer_col1, layer_col2)]
                        #child
                        elif (col1,col2) in childEdges:
                            conn_prob = childCol[(layer_col1, layer_col2)]
    
                        elif (col1,col2) in siblingEdges:
                            conn_prob = sibCol[(layer_col1,layer_col2)]
                        else:
                            conn_prob = 0.0

                    
                        for neuron_id_col1 in layerToNeuron[layer_id_col1]:
                            for neuron_id_col2 in layerToNeuron[layer_id_col2]:
                                if np.random.random() < conn_prob:
                                    neighbors[neuron_id_col2, neuron_id_col1] = np.random.random()*10**-3
                                    numConnections+=1
    print('Added {} connections'.format(numConnections - startConnections))
    return numConnections

def generateVisionNeurons(width=84, height=84,neurons_per_pixel=15):
    neurons = np.arange(width*height*neurons_per_pixel).reshape((width,height,neurons_per_pixel))
    return neurons

def connectVisionNeurons(Thalmus_levelToColumns, columnToLayer, layerToNeurons,vision_neurons, k, stride):
    last_level = len(Thalmus_levelToColumns)
    input_columns = Thalmus_levelToColumns[last_level-1]
    m,n = vision_neurons.shape[0], vision_neurons.shape[1]
    
    assert k < n
    assert (n - k) % stride == 0 and (m-k) % stride == 0

    num_hor, num_ver = int((n - k )/stride + 1), int((m-k)/stride + 1)
    num_windows = num_hor*num_ver

    assert len(input_columns) == num_windows
    
    total_input_neurons = 0
    for columnId in input_columns:
        layerId = columnToLayer[columnId][3] #4th layer
        total_input_neurons+= len(layerToNeurons[layerId])

    vision_adj = np.zeros((total_input_neurons, np.prod(vision_neurons.shape)),dtype=np.float16)
    vision_adj_idx_to_neuronId = {}
    inp_idx = 0
    columnIdx = 0
    for x in range(num_hor):
        for y in range(num_ver):
            vision_win = vision_neurons[y:y+k,x:x+k].flatten()
            layerId = columnToLayer[input_columns[columnIdx]][3] #4th layer
            inp_neurons = layerToNeurons[layerId]
            columnIdx+=1
            for inp in inp_neurons:
               vision_adj_idx_to_neuronId[inp_idx] = inp
               vision_adj[inp_idx, vision_win] = np.random.random() #initialize weights
               inp_idx+=1

    return vision_adj, vision_adj_idx_to_neuronId
            
def activateVisionNeurons(image, bins, vision_neurons, vision_spike_vector):
    ch = np.digitize(image, bins) - 1
    idx = np.array(list(np.ndindex((image.shape[0], image.shape[1]))))

    neurons = vision_neurons[idx[:,0],idx[:,1],ch.flatten()]
    vision_spike_vector*=0
    vision_spike_vector[neurons.flatten()] = 1
    return vision_spike_vector

def connectRegions(a_levelToColumns, b_levelToColumns, level_a, level_b, layer_a, layer_b,columnToLayer, layerToNeuron,neighbors,conn_prob = 0.05):
    #Send connections from columns in a to columns in b
    columns_a = a_levelToColumns[level_a] 
    columns_b = b_levelToColumns[level_b]
 
    numConnections = 0
    
    for col1 in columns_a:
        for col2 in columns_b:
            layer1 = columnToLayer[col1][layer_a] 
            layer2 = columnToLayer[col2][layer_b]
    
            for neuron_id_col1 in layerToNeuron[layer1]:
                for neuron_id_col2 in layerToNeuron[layer2]:
                    #Connect two neurons if applicable
                    if np.random.random() < conn_prob:
                        neighbors[neuron_id_col2, neuron_id_col1] = np.random.random()*10**-3
                        numConnections+=1
    print('Total connections added: {}'.format(numConnections))

                


def buildColumnToLayer(region_columns,prob_inhib_top_layer, prob_inhib_by_layer,count_by_ntype_by_layer, neuronId, layerId, columnId,columnToLayer,layerToNeuron, layerIdToLayerNum,neuronIdToNeuron, trans, num_layers=6):
    for columnId in region_columns:
        columnToLayer[columnId] = []
        for layer in range(num_layers):
            if layer == 0:
                neuronId,trans = generateLayer(prob_inhib_top_layer, count_by_ntype_by_layer[layer], neuronId, layerId, columnId,layerToNeuron,neuronIdToNeuron,trans)
            else:
                neuronId,trans = generateLayer(prob_inhib_by_layer[layer-1], count_by_ntype_by_layer[layer], neuronId, layerId, columnId,layerToNeuron,neuronIdToNeuron,trans)
       
            columnToLayer[columnId].append(layerId)
            layerIdToLayerNum[layerId] = layer
            layerId+=1

    return neuronId, layerId, columnId,trans



def getCountByNtypeByLayer(prob_by_ntype_by_layer, num_neurons_by_layer, num_layers=6):
    count_by_ntype_by_layer = []
    #get counts for each layer
    for layer in range(num_layers):
        count_by_ntype = {}
        prob_by_ntype_curr = prob_by_ntype_by_layer[layer]
        for k,v in prob_by_ntype_curr.items():
            count_by_ntype[k]= int(round(v*num_neurons_by_layer[layer] - 0.5))
        count_by_ntype_by_layer.append(count_by_ntype)
    return count_by_ntype_by_layer

def getNumNeuronsByLayer(mu_num_neurons_top, variance_param, mu_num_neurons_other, num_layers=6):
    num_neurons_by_layer  = []    
    for i in range(num_layers):
        if i == 0:
            num_neurons_by_layer.append(np.random.normal(mu_num_neurons_top, mu_num_neurons_top/variance_param))
        else:
            num_neurons_by_layer.append(np.random.normal(mu_num_neurons_other, mu_num_neurons_other/variance_param))
    return num_neurons_by_layer

def getNeuronBirthProb(birth_alpha_by_ntype, num_layers=6):
    prob_by_ntype_by_layer = []
    for layer in range(num_layers):
        prob_by_ntype_list = np.random.dirichlet([v for k,v in birth_alpha_by_ntype.items()])
        prob_by_ntype = {}
        for i, (k,v) in enumerate(birth_alpha_by_ntype.items()):
            prob_by_ntype[k] =prob_by_ntype_list[i]
        prob_by_ntype_by_layer.append(prob_by_ntype)

    return prob_by_ntype_by_layer

def getInhibNeuronBirthProb(inhib_alpha, num_layers=6):
    prob_inhib_by_layer = []
    for layer in range(num_layers-1):
        p_inhib = np.random.dirichlet(inhib_alpha)
        prob_inhib_by_layer.append(p_inhib[0])
    return prob_inhib_by_layer


def generateLayer(prob_inhibitory, count_by_ntype, neuronIdCount, layerId, columnId, layerToNeuron, neuronIdToNeuron,trans):
    layerToNeuron[layerId] = []
    for k,v in count_by_ntype.items():
        for _ in range(v):
            if np.random.random() < prob_inhibitory:
                trans.append(-1)
            else:
                trans.append(1)
            layerToNeuron[layerId].append(neuronIdCount)
            neuronIdCount+=1
            #print(neuronIdCount)
    return neuronIdCount,trans


def generateColumnTree(numRoots, numLevels, offspring_prob,parent_prob, sibling_prob, mu_offspring, columnIdCount=0):
    #edges = set()
    parents =   range(columnIdCount,columnIdCount +numRoots)
    offspring = []
    parentEdges, childEdges, siblingEdges, distEdges = set(), set(), set(), set()
    levelToColumns = {0: list(range(columnIdCount, columnIdCount  + numRoots))}
    #Add edges between roots
    for i in parents:
        for j in parents:
            if np.random.random() <= sibling_prob:
                #edges.add((i,j))
                siblingEdges.add((i,j))
    
    #create cortex graph level by level
    columnIdCount+=numRoots
    colId = parents[-1] + 1
    for level in range(numLevels-1):
        for parent in parents:
            #get number of offspring to generate
            numOffspring = np.random.poisson(mu_offspring,1)[0]
            #Generate offspring and add edges if appplicable
            if numOffspring > 0:
                for _ in range(numOffspring):
                    offspring.append(colId)
                    #edge from parent to child
                    if np.random.random() <= offspring_prob:
                        #edges.add((parent, colId))
                        parentEdges.add((parent, colId))
                    #edge from child to parent
                    if np.random.random() <= parent_prob:
                        #edges.add((colId, parent))
                        childEdges.add((colId, parent))
                    #increment columnId for next offspring
                    colId+=1
                    columnIdCount+=1
        #done creating offspring for this level, onto the next
        del parents
        levelToColumns[level+1] = offspring
        for off1 in offspring:
            for off2 in offspring:
                if np.random.random() <= sibling_prob:
                    siblingEdges.add((off1,off2))
                    
        parents = offspring
        offspring = []
        
    #n = colId
    #Generate adjacency matrix with parent offspring connections
    #adj = np.identity(n)
    #for (i,j) in edges:
    #    adj[i,j] = 1
        
    #graph = csr_matrix(adj)

    #dist_matrix = shortest_path(csgraph=graph, directed=True, return_predecessors=False)
    #Connect the rest of the nodes as per distance
    #for i in range(n):
    #    for j in range(j):
    #        if i != j and (i,j) not in edges:
    #            k = dist_matrix[i,j]
    #            if not np.isinf(k):
     #               p = (1-conn_prob_dis)**k*conn_prob_dis
     #               if np.random.random() < p:
    #                  adj[i,j] = 1
     #                   distEdges.add((i,j))
    
    
    
    return columnIdCount, levelToColumns, parentEdges, childEdges, siblingEdges
