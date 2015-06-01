#!/home/despoB/mb3152/anaconda/bin/python
import os
import sys
import pickle
import subprocess
import pandas as pd
import math
import time
from itertools import combinations
import numpy as np
import scipy
from scipy.stats.stats import pearsonr
from scipy import sparse, stats
from igraph import Graph, ADJ_UNDIRECTED
import glob
import nibabel as nib

#set this to where your subjects are located. The SUBJECT will be replaced with your subject ID
# 1/0
class brain_graph:
	def __init__(self, graph, communities,igraph_object):
		self.graph = graph
		self.community_array = communities
		self.communities = array_to_listlist(communities)
		degree = np.zeros((graph.vcount()))
		size_array = np.zeros(graph.vcount())
		for node_idx,node in enumerate(range(graph.vcount())):
			size_array[node_idx] = igraph_object.sizes()[communities[node]]
		self.community_size = size_array
		for node_idx,node1 in enumerate(range(graph.vcount())):
			if graph.degree(node_idx) == 0.:
				continue
			node_degree = 0.
			for node2 in range(graph.vcount()):
				node_degree = node_degree + max(graph[node1,node2],graph[node2,node1])
			degree[node_idx] = node_degree
		self.weighted_degree = degree
		node_degree_by_community = np.zeros((graph.vcount(),len(self.communities)))
		for node_idx,node1 in enumerate(range(graph.vcount())):
			node_degree_for_community = np.zeros(len(self.communities))
			for comm_idx,comm in enumerate(self.communities):
				comm_total_degree = 0.
				for node2 in comm:
					comm_total_degree = comm_total_degree + max(graph[node1,node2],graph[node2,node1])
				node_degree_by_community[node_idx,comm_idx] = comm_total_degree
		self.node_degree_by_community = node_degree_by_community
		pc_array = np.zeros(graph.vcount())
		for node in range(graph.vcount()):
		    node_degree = self.weighted_degree[node]
		    if node_degree == 0.0: 
		        pc_array[node]= np.nan
		        continue    
		    pc = 0.0
		    for comm_degree in self.node_degree_by_community[node]:
		        pc = pc + ((float(comm_degree)/float(node_degree))**2)
		    pc = 1 - pc
		    pc_array[int(node)] = float(pc)
		self.pc = pc_array

def load_subject_time_series(subject_dir):
	"""
	returns a 4d array of the epi files.
	loads original file in data_dir 
	or a resampled file if mm = 3 or 4
	"""
	files = glob.glob(subject_dir)
	for block,img_file in enumerate(files):
		print 'loading: ' + str(img_file)
		reorient(image=img_file, orientation='RPI')
		if block == 0:
			epi_data = nib.load(img_file).get_data().astype('float32') # move up
			continue 
		new_epi_data = nib.load(img_file).get_data().astype('float32')
		epi_data = np.concatenate((epi_data,new_epi_data),axis =3)
	return epi_data

def time_series_to_matrix(subject_time_series,parcel,voxel=False,low_tri_only=False,sparse=False,fisher=False,out_file='/home/despoB/mb3152/voxel_matrices/'):
	"""
	Makes correlation matrix from parcel
	If voxel == true, masks the subject_time_series with the parcel 
	and runs voxel correlation on those voxels. Ignores touching voxels.
	Only fills the lower triangle! Saves memory.
	"""
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
		g = scipy.sparse.lil_matrix((len(ts),len(ts)))	
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
		if sparse == False:
			g = g.todense()
	return g

def relabel_part(lp):
	# 0 index is for nodes without a module
	non_empty_modules = []
	new_lp= []
	lp = lp + 1
	for i in range(1,int(max(lp)+1)):
		if len(lp[lp==i]) > 0.:
			non_empty_modules.append(i)
	translation_dict = {}
	translation_dict[0] = 0
	for i,module in enumerate(non_empty_modules):
		translation_dict[module] = i + 1
	for i in lp:
		new_lp.append(translation_dict[i])
	return np.array(new_lp)

def scipy_to_igraph(matrix, cost, directed=False):
	"""
	must be only the lower triangle!
	"""
	tot_edges = (((matrix.shape[0]))*(((matrix.shape[0]))-1))/2.    
	df = pd.DataFrame()
	df['Sources'],df['Targets'] = matrix.nonzero()
	df['Weights'] = matrix[df.Sources,df.Targets].data[0]
	del matrix
	n_nonzero = cost * tot_edges
	df.sort(columns='Weights',ascending=False,inplace=True)
	df = df.reset_index()
	threshold = df.Weights[int(np.round(n_nonzero))+1]
	df = df[df.Weights >= threshold]
	return Graph(zip(df.Sources.values, df.Targets.values), directed=directed, edge_attrs={'weight': df.Weights.values})

def threshold_matrix(matrix,cost,check_tri=True,lower_tri_only=False):
	c_cost_int = 100 - (cost*100)
	if check_tri == True:
		if np.sum(np.triu(matrix)) == 0.0 or np.sum(np.tril(matrix)) == 0.0:
			c_cost_int = 100 - ((cost/2.)*100)
	matrix[matrix<np.percentile(matrix,c_cost_int,interpolation='nearest')] = 0
	return matrix

def make_graph(subject,num_nodes=1000,atlas='ward'):
	t1 = time.time()
	if atlas == 'ward':
		parcel = glob.glob('/home/despoB/mb3152/nodeless_networks/nodes/nodes_%s_%s.nii'%(subject,num_nodes))[0]
	else:
		parcel = '/home/despoB/mb3152/nodeless_networks/real_atlases/4mm_%s_template.nii'%(atlas)
	subject_time_series_data = load_sub_data(subject,mm=4)
	parcel = nib.load(parcel).get_data()
	matrix = time_series_to_matrix(subject_time_series=subject_time_series_data,voxel=False,parcel=parcel)
	del subject_time_series_data
	cost = 0.25
	temp_matrix = np.zeros((matrix.shape[0],matrix.shape[0]))
	pc_matrix = threshold_matrix(matrix,.1)
	pc_graph = Graph.Weighted_Adjacency(pc_matrix.tolist(),mode= ADJ_UNDIRECTED,attr="weight")
	del pc_matrix
	while cost > 0.009:
		print cost
		matrix = threshold_matrix(matrix,cost)
		g = Graph.Weighted_Adjacency(matrix.tolist(),mode= ADJ_UNDIRECTED,attr="weight")
		g = g.community_infomap(edge_weights='weight')
		part_object = partition_no_pc(g.graph,g.membership,g)
		real_nodes = []
		for node in range(part_object.graph.vcount()):
			if part_object.graph.degree([node]) > 0:
				if part_object.community_size[node] > 10:
 					real_nodes.append(node)
		same_edges = []
		diff_edges = []
		for edge in combinations(real_nodes,2):
			if part_object.community_array[edge[0]] == part_object.community_array[edge[1]]:
				same_edges.append(edge)
			else:
				diff_edges.append(edge)
		for i,edge in enumerate(same_edges):
			temp_matrix[edge[0],edge[1]] = 1
			temp_matrix[edge[1],edge[0]] = 1
		for i,edge in enumerate(diff_edges):
			temp_matrix[edge[0],edge[1]] = 0
			temp_matrix[edge[1],edge[0]] = 0
		if cost > .1:
			cost = cost - .01
		elif cost > 0.05:
			cost = cost - 0.005
		else:
			cost = cost - 0.001
	g = Graph.Weighted_Adjacency(temp_matrix.tolist(),mode= ADJ_UNDIRECTED,attr="weight")
	g = g.community_infomap(edge_weights='weight')
	part_object = partition(pc_graph,g.membership,g)
	part_object.c_matrix = temp_matrix
	if atlas == 'ward':
		fill_ward_voxel_matrix(part_object,parcel,subject,cost,num_nodes)
	else:
		fill_real_voxel_matrix(part_object,parcel,subject,cost,num_nodes,atlas)
	print time.time() - t1
	pickle.dump(part_object, open( "/home/despoB/mb3152/nodeless_networks/graphs/partition_%s_%s_%s.p" %(subject,num_nodes,atlas), "wb" ))