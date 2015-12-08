#!/home/despoB/mb3152/anaconda/bin/python
import matlab.engine
import os
import sys
import pickle
import glob
import numpy as np
import numpy.testing as npt
from scipy.stats.stats import pearsonr
from sklearn.feature_extraction import image
from igraph import Graph, ADJ_UNDIRECTED, VertexClustering
import nibabel as nib
from itertools import combinations
import pandas as pd

def load_matlab_toolbox(matlab_library):
	eng = matlab.engine.start_matlab()
	if matlab_library == 'bct':
		eng.addpath('/home/despoB/mb3152/brain_graphs/bct/')
	if matlab_library == 'dcc':
		eng.addpath('/home/despoB/mb3152/brain_graphs/dcc/DCCcode/')
	return eng

def load_graph(path_to_graph):
    f = open('%s' %(path_to_graph),'r')
    return pickle.load(f)

def save_graph(path_to_graph):
    f = open('%s' %(path_to_graph),'r')
    return pickle.load(f)

class brain_graph:
	def __init__(self, VC):
		assert (np.unique(VC.membership) == range(len(VC.sizes()))).all()
		node_degree_by_community = np.zeros((VC.graph.vcount(),len(VC.sizes())),dtype=np.float64)
		for node1 in range(VC.graph.vcount()):
			for comm_idx in (np.unique(VC.membership)):
				comm_total_degree = 0.
				for node2 in np.argwhere(np.array(VC.membership)==comm_idx).reshape(-1):
					eid = VC.graph.get_eid(node1,node2,error=False)
					if eid == - 1:
						continue
					weight = VC.graph.es[eid]["weight"]
					comm_total_degree = comm_total_degree + weight
				node_degree_by_community[node1,comm_idx] = comm_total_degree
		# node_degree_by_community[np.argwhere(np.max(node_degree_by_community,axis=1)==0),:] = np.nan
		pc_array = np.zeros(VC.graph.vcount())
		for node in range(VC.graph.vcount()):
		    node_degree = VC.graph.strength(node,weights='weight')
		    if node_degree == 0.0: 
		        pc_array[node]= np.nan
		        continue    
		    pc = 0.0
		    for idx,comm_degree in enumerate(node_degree_by_community[node]):
		        pc = pc + ((float(comm_degree)/float(node_degree))**2)
		    pc = 1.0 - pc
		    pc_array[int(node)] = float(pc)
		self.pc = pc_array
		wmd_array = np.zeros(VC.graph.vcount())
		for comm_idx in range(len(VC.sizes())):
			comm = np.argwhere(np.array(VC.membership)==comm_idx).reshape(-1)
			comm_std = np.std(node_degree_by_community[comm,comm_idx],dtype=np.float64)
			comm_mean = np.mean(node_degree_by_community[comm,comm_idx],dtype=np.float64)
			for node in comm:
				node_degree = VC.graph.strength(node,weights='weight')
				comm_node_degree = node_degree_by_community[node,comm_idx]
				if node_degree == 0.0:
					wmd_array[node] = np.nan
					continue
				if comm_std == 0.0:
					assert comm_node_degree == comm_mean
					wmd_array[node] = 0.0
					continue
				wmd_array[node] = np.divide((np.subtract(comm_node_degree,comm_mean)),comm_std)
		self.wmd = wmd_array
		self.community = VC
		self.node_degree_by_community = node_degree_by_community
		self.matrix = np.array(self.community.graph.get_adjacency(attribute='weight').data)

def between_community_centrality(brain_graph,start_end_nodes,weight=False):
	bcc_array = np.zeros(brain_graph.community.graph.vcount())
	for node1 in range(brain_graph.community.graph.vcount()):
		for node2 in range(brain_graph.community.graph.vcount()):
			if node1 not in start_end_nodes or node2 not in start_end_nodes:
				continue
			if brain_graph.community.membership[node1] == brain_graph.community.membership[node2]:
				continue
			if weight == True:
				indices = np.ix_(np.argwhere(np.array(brain_graph.community.membership)==x).reshape(-1),np.argwhere(np.array(brain_graph.community.membership)==y).reshape(-1))
				path_weight = 1 - np.nanmean(brain_graph.matrix[indices])
			else:
				path_weight = 1.
			paths = brain_graph.community.graph.get_shortest_paths(node1,node2,weights='weight',output ='vpath')[0][1:-1]
			for p in paths:
				bcc_array[p] = bcc_array[p] + (1*path_weight)
	return bcc_array

def coupling(data,window):
    """
    creates a functional coupling metric from 'data'
    using the multiplication of temporal derivatives. 
    data: should be organized in 'time x nodes' matrix
    smooth: smoothing parameter for dynamic coupling score
    taken from: https://github.com/macshine/coupling/blob/master/coupling.py
    """
    
    #define variables
    [tr,nodes] = data.shape
    der = tr-1
    td = np.zeros((der,nodes))
    td_std = np.zeros((der,nodes))
    data_std = np.zeros(nodes)
    mtd = np.zeros((der,nodes,nodes))
    sma = np.zeros((der,nodes*nodes))
    
    #calculate temporal derivative
    for i in range(0,nodes):
        for t in range(0,der):
            td[t,i] = data[t+1,i] - data[t,i]
    
    
    #standardize data
    for i in range(0,nodes):
        data_std[i] = np.std(td[:,i])
    
    td_std = td / data_std
   
   
    #functional coupling score
    for t in range(0,der):
        for i in range(0,nodes):
            for j in range(0,nodes):
                mtd[t,i,j] = td_std[t,i] * td_std[t,j]


    #temporal smoothing
    temp = np.reshape(mtd,[der,nodes*nodes])
    sma = pd.rolling_mean(temp,window)
    sma = np.reshape(sma,[der,nodes,nodes])
    
    return (mtd, sma)

def make_image(atlas_path,image_path,values,fill=False):
	image = nib.load(atlas_path)
	image_data = image.get_data()
	shape = image_data.shape
	value_data = image_data.copy()
	for ix,i in enumerate(values):
		value_data[image_data==ix+1] = i
	image_data[:,:,:,] = value_data[:,:,:,]
	nib.save(image,image_path)

def clean_up_membership(partition,matrix,min_community_size):
	for min_community_size in range(2,min_community_size+1):
		small_nodes = []
		small_communities = []
		membership = []
		for node in range(len(partition.membership)):
			if partition.sizes(partition.membership[node])[0] < min_community_size:
				small_nodes.append(node)
				small_communities.append(partition.membership[node])
		for node in range(len(partition.membership)):
			if node not in small_nodes:
				membership.append(partition.membership[node])
				continue
			community_weights = []
			for community in range(len(partition.sizes())):
				if community not in small_communities:
					community_weights.append(np.nansum(matrix[node][np.argwhere(np.array(partition.membership) == community)]))
				else:
					community_weights.append(0.0)
			community_weights = np.array(community_weights)
			community_weights[np.isnan(community_weights)] = 0.0
			if np.nanmax(community_weights) == 0.0:
				membership.append(partition.membership[node])
				continue
			membership.append(np.argmax(community_weights))
		membership = np.array(membership)
		temp_partition = VertexClustering(partition.graph, membership=membership)
		empty = np.argwhere(np.array(temp_partition.sizes())==0).reshape(-1)
		diff = 0
		for e in empty:
			over = np.argwhere(membership > (e - diff)).reshape(-1)
			for m in over:
				membership[m] = membership[m] - 1
			diff = diff + 1
		partition = VertexClustering(partition.graph, membership=membership)
	return membership

def load_graph(path_to_graph):
	f = open(path_to_graph,'r')
	return pickle.load(f)

def save_graph(path_to_graph,partition):
    f = open(path_to_graph,'w+')
    pickle.dump(partition,f)
    f.close()

def load_subject_time_series(subject_path,scrub_mm=False):
	"""
	returns a 4d array of the subject_time_series files.
	loads original file in subject_path
	"""
	files = glob.glob(subject_path)
	for block,img_file in enumerate(files):
		print 'loading: ' + str(img_file)
		if scrub_mm != False:
			dis_file = np.loadtxt(img_file.split('functional_mni')[0] + 'frame_wise_displacement/' + img_file.split('functional_mni')[1].split('/')[1] + '/FD.1D')
			remove_array = np.zeros(len(dis_file))
			for i,f in enumerate(dis_file):
				if f > scrub_mm:
					remove_array[i] = True
					if i == 0:
						remove_array[i+1] = True
						continue
					if i == len(dis_file)-1:
						remove_array[i-1] = True
						continue
					remove_array[i-1] = True
					remove_array[i+1] = True
			remove_array[remove_array==0.0] = False
		if block == 0:
			subject_time_series_data = nib.load(img_file).get_data().astype('float32')
			if scrub_mm != False:
				subject_time_series_data = np.delete(subject_time_series_data,np.where(remove_array==True),axis=3)
			continue
		new_subject_time_series_data = nib.load(img_file).get_data().astype('float32')
		if scrub_mm != False:
			new_subject_time_series_data = np.delete(new_subject_time_series_data,np.where(remove_array==True),axis=3)
		subject_time_series_data = np.concatenate((subject_time_series_data,new_subject_time_series_data),axis =3)
	return subject_time_series_data

def count_scrubbed_frames(subject_path,scrub_mm=0.2):
	"""
	returns a 4d array of the subject_time_series files.
	loads original file in subject_path
	"""
	files = glob.glob(subject_path)
	kept_frames = []
	all_frames = []
	for block,img_file in enumerate(files):
		dis_file = np.loadtxt(img_file.split('functional_mni')[0] + 'frame_wise_displacement/' + img_file.split('functional_mni')[1].split('/')[1] + '/FD.1D')
		kept_frames.append(len(dis_file)-len(dis_file[dis_file>=scrub_mm]))
		all_frames.append(len(dis_file))
	return kept_frames,all_frames

def times_series_to_interp_matrix(subject_time_series,parcel_path,interpolation_points):
	parcel = nib.load(parcel_path).get_data()
	ts = dict()
	for i in range(np.max(parcel)):
		final_nodes_ts = []
		node_ts = np.mean(subject_time_series[parcel==i+1],axis = 0)
		for tp in node_ts:
			final_nodes_ts.append(tp)
			for x in range(interpolation_points):
				final_nodes_ts.append(np.nan)
		ts[i] = final_nodes_ts
	ts = pd.DataFrame(ts)

def time_series_to_ewmf_matrix(subject_time_series,parcel_path,window_size,out_file=None):
	"""
	runs exponentially weighted moment functions via Pandas
	"""
	parcel = nib.load(parcel_path).get_data()
	ts = dict()
	for i in range(np.max(parcel)):
		ts[i] = np.mean(subject_time_series[parcel==i+1],axis = 0)
	ts = pd.DataFrame(ts)
	matrix = pd.ewmcorr(ts,span=window_size)
	if out_file != None:
		np.save(out_file,np.array(matrix))
	else:
		return np.array(matrix)

def time_series_to_dcc_matrix(subject_time_series,parcel_path,out_file):
	from scipy.stats.mstats import zscore as z_score
	eng = matlab.engine.start_matlab()
	"""
	runs DCC method from M Lindquist
	"""
	parcel = nib.load(parcel_path).get_data()
	ts = np.zeros((np.max(parcel),subject_time_series.shape[3]))
	for i in range(np.max(parcel)):
		ts[i,:] = z_score(np.mean(subject_time_series[parcel==i+1],axis = 0))
	ts = ts.swapaxes(0,1)
	matrices = eng.mvDCC(matlab.double(ts.tolist()))
	np.save(out_file,np.array(matrices))

def time_series_to_matrix(subject_time_series,parcel_path,voxel=False,fisher=False,out_file=None):
	"""
	Makes correlation matrix from parcel
	If voxel == true, masks the subject_time_series with the parcel 
	and runs voxel correlation on those voxels.
	"""
	parcel = nib.load(parcel_path).get_data()
	if voxel == True:
		flat_parcel = parcel.reshape(-1)
		g = np.memmap(out_file, dtype='float32', mode='r+', shape=(len(flat_parcel),len(flat_parcel)))
		subject_time_series = subject_time_series.reshape(-1,subject_time_series.shape[-1])
		subject_time_series = subject_time_series[np.argwhere(flat_parcel>0)]
		subject_time_series = subject_time_series.reshape((subject_time_series.shape[0],subject_time_series.shape[-1]))
		g = np.corrcoef(subject_time_series)
	else:
		g = np.zeros((np.max(parcel),subject_time_series.shape[-1]))
		for i in range(np.max(parcel)):
			g[i,:] = np.nanmean(subject_time_series[parcel==i+1],axis = 0)
		g = np.corrcoef(g)
		if fisher == True:
			g = np.arctanh(g)
		if out_file != None:
			np.save(out_file,g)
	return g

def partition_avg_costs(matrix,costs,min_community_size):
	final_edge_matrix = matrix.copy()
	final_matrix = []
	for cost in costs:
		graph = matrix_to_igraph(matrix.copy(),cost)
		partition = graph.community_infomap(edge_weights='weight')
		final_matrix.append(community_matrix(partition.membership,min_community_size))
	final_graph = matrix_to_igraph(np.nanmean(final_matrix,axis=0),cost=1.)
	partition = graph.community_infomap(edge_weights='weight')
	return partition.membership

def partition_exponentially_weight(matrix,num_communities,min_community_size,avg=False):
	community_len = 0
	exp = 1
	community_lenths = []
	exponents = []
	while community_len < num_communities:
		exp = exp + .1
		temp_matrix = matrix.copy()
		temp_matrix = temp_matrix**exp
		graph = matrix_to_igraph(temp_matrix,cost=1.)
		partition = graph.community_infomap(edge_weights='weight')
		membership = np.array(partition.sizes())
		community_len = len(membership[membership>min_community_size])
		community_lenths.append(community_len)
		exponents.append(exp)
		print 'Exponent: ' + str(exp) + ', Communities: ' + str(community_len)
		if np.max(community_lenths) > np.max(community_lenths[-10:]):
			exp = exponents[np.argmax(community_lenths)]
			print 'breaking early with exponent of: ' + str(exp)
			break
	if avg == True:
		exponents = np.linspace(exp-.1,exp+.1,num=1000)
		final_matrix = []
		for avg_exp in exponents:
			temp_matrix = matrix.copy()
			temp_matrix = temp_matrix**avg_exp
			graph = matrix_to_igraph(temp_matrix,cost=1.)
			partition = graph.community_infomap(edge_weights='weight')
			final_matrix.append(community_matrix(partition.membership,min_community_size))
		final_graph = matrix_to_igraph(np.nanmean(final_matrix,axis=0),cost=1.)
		partition = graph.community_infomap(edge_weights='weight')
	return np.array(partition.membership),exp

def matrix_to_igraph(matrix,cost,binary=False,check_tri=True,interpolation='midpoint',normalize=False):
	matrix = threshold(matrix,cost,binary,check_tri,interpolation,normalize)
	g = Graph.Weighted_Adjacency(matrix.tolist(),mode= ADJ_UNDIRECTED,attr="weight")
	if np.diff([cost,g.density()]) > .005:
		print 'Density not %s! Did you want: ' %(cost)+ str(g.density()) + ' ?' 
	return g

def threshold(matrix,cost,binary=False,check_tri=True,interpolation='midpoint',normalize=False):
	matrix[np.isnan(matrix)] = 0.0
	matrix[matrix<0.0] = 0.0
	np.fill_diagonal(matrix,0.0)
	c_cost_int = 100-(cost*100)
	if check_tri == True:
		if np.sum(np.triu(matrix)) == 0.0 or np.sum(np.tril(matrix)) == 0.0:
			c_cost_int = 100.-((cost/2.)*100.)
	if c_cost_int > 0:
		matrix[matrix<np.percentile(matrix,c_cost_int,interpolation=interpolation)] = 0.
	if binary == True:
		matrix[matrix>0] = 1
	if normalize == True:
		matrix = matrix/np.sum(matrix)
	return matrix

def community_matrix(membership,min_community_size):
	membership = np.array(membership).reshape(-1)
	final_matrix = np.zeros((len(membership),len(membership)))
	final_matrix[:] = np.nan
	connected_nodes = []
	for i in np.unique(membership):
		if len(membership[membership==i]) >= min_community_size:
			for c in np.array(np.where(membership==i)):
				for n in c:
					connected_nodes.append(int(n))
	community_edges = []
	between_community_edges = []
	connected_nodes = np.array(connected_nodes)
	for edge in combinations(connected_nodes,2):
		if membership[edge[0]] == membership[edge[1]]:
			community_edges.append(edge)
		else:
			between_community_edges.append(edge)
	for edge in community_edges:
		final_matrix[edge[0],edge[1]] = 1
		final_matrix[edge[1],edge[0]] = 1
	for edge in between_community_edges:
		final_matrix[edge[0],edge[1]] = 0
		final_matrix[edge[1],edge[0]] = 0
	return final_matrix

def multi_slice_community(matrix,cost,out_file,omega=.1,gamma=1.0):
	"""
	matrix: a matrix with the first dimenstion as time points.

	resturns community detection for each time point as similar matrix
	"""
	eng = matlab.engine.start_matlab()
	eng.addpath('/home/despoB/mb3152/brain_graphs')
	shape = matrix.shape
	matlab_matrix = []
	print 'Converting Matrix for MATLAB'
	for i in range(matrix.shape[0]):
		matlab_matrix.append(matlab.double(threshold(matrix[i,:,:],cost).tolist()))
	c_matrix = np.array(eng.genlouvain(matlab_matrix,1000,1,1,1,omega,gamma))
	c_matrix = c_matrix.reshape(shape[:2])
	np.save(out_file,c_matrix)


def average_recursive_network_partition(parcel_path=None,subject_path=None,matrix=None,graph_cost=.1,max_cost=.25,min_cost=0.05,min_community_size=5,min_weight=1.):
	"""
	subject_past: list of paths to subject file or files

	Combines network partitions across costs (Power et al, 2011)
	Starts at max_cost, finds partitions that nodes are in,
	slowly decreases density to find smaller partitions, but keeps 
	information (from higher densities) about nodes that become disconnected.

	Runs nodal roles on one cost (graph_cost), but with final partition.

	Returns brain_graph object.
	"""
	
	if matrix == None:
		subject_time_series_data = load_subject_time_series(subject_path)
		matrix = time_series_to_matrix(subject_time_series=subject_time_series_data,voxel=False,parcel_path=parcel_path)
		matrix = np.nanmean(matrix,axis=0)
		matrix[matrix<0] = 0.0
		np.fill_diagonal(matrix,0)
	matrix[matrix<0] = 0.0
	np.fill_diagonal(matrix,0)
	final_edge_matrix = matrix.copy()
	final_matrix = []
	cost = max_cost
	final_graph = matrix_to_igraph(matrix.copy(),cost=graph_cost)
	while True:
		temp_matrix = np.zeros((matrix.shape[0],matrix.shape[0]))
		graph = matrix_to_igraph(matrix,cost=cost)
		partition = graph.community_infomap(edge_weights='weight')
		community_matrix(partition.community.membership,min_community_size)
		if cost < min_cost:
			break
		if cost <= .05:
			cost = cost - 0.001
			continue
		if cost <= .15:
			cost = cost - 0.01
			continue
		if cost >= .3:
			cost = cost - .05
			continue
		if cost > .15:
			cost = cost - 0.01
			continue
	graph = matrix_to_igraph(final_matrix*final_edge_matrix,cost=1.)
	partition = graph.community_infomap(edge_weights='weight')
	return brain_graph(VertexClustering(final_graph, membership=partition.membership))

def recursive_network_partition(parcel_path=None,subject_path=None,matrix=None,graph_cost=.1,max_cost=.25,min_cost=0.05,min_community_size=5,min_weight=1.):
	"""
	subject_past: list of paths to subject file or files

	Combines network partitions across costs (Power et al, 2011)
	Starts at max_cost, finds partitions that nodes are in,
	slowly decreases density to find smaller partitions, but keeps 
	information (from higher densities) about nodes that become disconnected.

	Runs nodal roles on one cost (graph_cost), but with final partition.

	Returns brain_graph object.
	"""
	
	if matrix == None:
		subject_time_series_data = load_subject_time_series(subject_path)
		matrix = time_series_to_matrix(subject_time_series=subject_time_series_data,voxel=False,parcel_path=parcel_path)
		matrix = np.nanmean(matrix,axis=0)
		matrix[matrix<0] = 0.0
		np.fill_diagonal(matrix,0)
	matrix[matrix<0] = 0.0
	np.fill_diagonal(matrix,0)
	final_edge_matrix = matrix.copy()
	final_matrix = np.zeros(matrix.shape)
	cost = max_cost
	final_graph = matrix_to_igraph(matrix.copy(),cost=graph_cost)
	while True:
		temp_matrix = np.zeros((matrix.shape[0],matrix.shape[0]))
		graph = matrix_to_igraph(matrix,cost=cost)
		partition = graph.community_infomap(edge_weights='weight')
		connected_nodes = []
		for node in range(partition.graph.vcount()):
			if partition.graph.strength(node,weights='weight') > min_weight:
				if partition.sizes()[partition.membership[node]] > min_community_size:
 					connected_nodes.append(node)
		community_edges = []
		between_community_edges = []
		for edge in combinations(connected_nodes,2):
			if partition.membership[edge[0]] == partition.membership[edge[1]]:
				community_edges.append(edge)
			else:
				between_community_edges.append(edge)
		for edge in community_edges:
			final_matrix[edge[0],edge[1]] = 1
			final_matrix[edge[1],edge[0]] = 1
		for edge in between_community_edges:
			final_matrix[edge[0],edge[1]] = 0
			final_matrix[edge[1],edge[0]] = 0
		if cost < min_cost:
			break
		if cost <= .05:
			cost = cost - 0.001
			continue
		if cost <= .15:
			cost = cost - 0.01
			continue
		if cost >= .3:
			cost = cost - .05
			continue
		if cost > .15:
			cost = cost - 0.01
			continue
	graph = matrix_to_igraph(final_matrix*final_edge_matrix,cost=1.)
	partition = graph.community_infomap(edge_weights='weight')
	return brain_graph(VertexClustering(final_graph, membership=partition.membership))
