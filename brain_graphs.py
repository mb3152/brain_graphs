#!/home/despoB/mb3152/anaconda/bin/python
import os
import sys
import pickle
import glob
import numpy as np
import numpy.testing as npt
from scipy.stats.stats import pearsonr
from igraph import Graph, ADJ_UNDIRECTED, VertexClustering
import nibabel as nib
from itertools import combinations

class brain_graph:
	def __init__(self, VC):
		node_degree_by_community = np.zeros((VC.graph.vcount(),len(VC.sizes())))
		for node1 in range(VC.graph.vcount()):
			for comm_idx in (np.unique(VC.membership)):
				comm_total_degree = 0.
				for node2 in np.argwhere(np.array(VC.membership)==comm_idx).reshape(-1):
					comm_total_degree = comm_total_degree + max(VC.graph[node1,node2],VC.graph[node2,node1])
				node_degree_by_community[node1,comm_idx] = comm_total_degree
		node_degree_by_community[np.argwhere(np.max(node_degree_by_community,axis=1)==0),:] = np.nan
		pc_array = np.zeros(VC.graph.vcount())
		for node in range(VC.graph.vcount()):
		    node_degree = VC.graph.strength(node,weights='weight')
		    if node_degree == 0.0: 
		        pc_array[node]= np.nan
		        continue    
		    pc = 0.0
		    for comm_degree in node_degree_by_community[node]:
		        pc = pc + ((float(comm_degree)/float(node_degree))**2)
		    pc = 1 - pc
		    pc_array[int(node)] = float(pc)
		self.pc = pc_array
		wmd_array = np.zeros(VC.graph.vcount())
		for comm_idx in (np.unique(VC.membership)):
			comm = np.argwhere(np.array(VC.membership)==comm_idx).reshape(-1)
			comm_std = np.nanstd(node_degree_by_community[comm,comm_idx])
			comm_mean = np.nanmean(node_degree_by_community[comm,comm_idx])
			for node in comm:
				node_degree = VC.graph.strength(node,weights='weight')
				comm_node_degree = node_degree_by_community[node,comm_idx]
				if node_degree == 0.0:
					wmd_array[node] = np.nan
					continue
				if comm_std == 0.0:
					wmd_array[node] = (comm_node_degree-comm_mean)
					continue	
				wmd_array[node] = (comm_node_degree-comm_mean) / comm_std
		self.wmd = wmd_array
		self.community = VC
		self.node_degree_by_community = node_degree_by_community

def load_subject_time_series(subject_path):
	"""
	returns a 4d array of the subject_time_series files.
	loads original file in subject_path
	"""
	files = glob.glob(subject_path)
	for block,img_file in enumerate(files):
		print 'loading: ' + str(img_file)
		if block == 0:
			subject_time_series_data = nib.load(img_file).get_data().astype('float32') # move up
			continue 
		new_subject_time_series_data = nib.load(img_file).get_data().astype('float32')
		subject_time_series_data = np.concatenate((subject_time_series_data,new_subject_time_series_data),axis =3)
	# if 
	return subject_time_series_data

def time_series_to_matrix(subject_time_series,parcel_path,voxel=False,low_tri_only=False,fisher=False,out_file='/home/despoB/mb3152/voxel_matrices/'):
	"""
	Makes correlation matrix from parcel
	If voxel == true, masks the subject_time_series with the parcel 
	and runs voxel correlation on those voxels. Ignores touching voxels.
	low_tri_only: only fills the lower triangle to save memory.
	"""
	parcel = nib.load(parcel_path).get_data()
	if voxel == True:
		flat_parcel = parcel.reshape(-1)
		shape = parcel.shape
		distance_matrix = image.grid_to_graph(shape[0],shape[1],shape[2]).tocsr()
		g = np.memmap(out_file, dtype='float32', mode='r+', shape=(len(flat_parcel),len(flat_parcel)))
		subject_time_series = subject_time_series.reshape(-1,subject_time_series.shape[-1])
		edges = np.argwhere(flat_parcel>0)
		for edge in combinations(edges,2):
			edge = [edge[0][0],edge[1][0]]
			i2 = min(edge)
			i1 = max(edge)
			if flat_parcel[i2] == 0:
				continue
			if flat_parcel[i1] == 0:
				continue
			if distance_matrix[i2,i1] == 1:
				continue
			if distance_matrix[i1,i2] == 1:
				print 'second check on distance matrix used'
				continue
			g[i1,i2]= pearsonr(subject_time_series[i1,:,],subject_time_series[i2,:,])[0]
	else:
		ts = dict()
		for i in range(np.max(parcel)):
			ts[i] = np.mean(subject_time_series[parcel==i+1],axis = 0)
		g = np.zeros((len(ts),len(ts)))	
		for edge in combinations(ts.keys(),2):
			i2 = min(edge)
			i1 = max(edge)
			correlation = pearsonr(ts[i2],ts[i1])[0]
			if correlation > 0.0:
				g[i1,i2] = correlation
				if low_tri_only == False:
					g[i2,i1] = correlation
		if fisher == True:
			g = np.arctanh(g)
	return g

def matrix_to_igraph(matrix,cost,binary=False,check_tri=True):
	matrix[np.isnan(matrix)] = 0
	c_cost_int = 100-(cost*100)
	if check_tri == True:
		if np.sum(np.triu(matrix)) == 0.0 or np.sum(np.tril(matrix)) == 0.0:
			c_cost_int = 100.-((cost/2.)*100.)
	if c_cost_int > 0:
		matrix[matrix<np.percentile(matrix,c_cost_int,interpolation='nearest')] = 0
	if binary == True:
		matrix[matrix>0] = 1
	g = Graph.Weighted_Adjacency(matrix.tolist(),mode= ADJ_UNDIRECTED,attr="weight")
	# npt.assert_almost_equal(g.density(), cost, decimal=2, err_msg='Error while thresholding matrix', verbose=True)
	return g

def recursive_network_partition(subject_path,parcel_path,graph_cost=.1,max_cost=.5,min_cost=0.01,min_community_size=5):
	"""
	Combines network partitions across costs (Power et al, 2011)
	Starts at max_cost, finds partitions that nodes are in,
	slowly decreases density to find smaller partitions, but keeps 
	information (from higher densities) about nodes that become disconnected.

	Runs nodal roles on one cost (graph_cost), but with final partition.

	Returns brain_graph object.
	"""
	subject_time_series_data = load_subject_time_series(subject_path)
	matrix = time_series_to_matrix(subject_time_series=subject_time_series_data,voxel=False,parcel_path=parcel_path)
	final_matrix = np.zeros(matrix.shape)
	# del subject_time_series_data
	cost = max_cost
	final_graph = matrix_to_igraph(matrix.copy(),cost=graph_cost)
	while True:
		graph = matrix_to_igraph(matrix,cost=cost)
		partition = graph.community_infomap(edge_weights='weight')
		connected_nodes = []
		for node in range(partition.graph.vcount()):
			if partition.graph.strength(node,weights='weight') > 0.:
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
		cost = cost - 0.01
	graph = matrix_to_igraph(matrix,cost=1.)
	partition = graph.community_infomap(edge_weights='weight')
	return brain_graph(VertexClustering(final_graph, membership=partition.membership)