# brain_graphs
Graph theory analysis of resting-state fMRI data. Just a light class to summarize igraph (python) community detection object and the igraph graph object. Calculates Participation Coefficient(PC) and Within-Module-Degree Z-Score(WMD). 

All graphs are weighted unless passed a binarized Matrix.

Requires two non-standard (i.e., not with anaconda distribution) libraries:

python-igraph $pip install python-igraph

nibabel $pip install nibabel

Example 1:

~Load EPI and reduce to a time series of nodes

subject_time_series = load_subject_time_series(subject_path='path/to/where/epi/files/live/*')

~Turn that into a correlation matrix based on parcellation

matrix = time_series_to_matrix(subject_time_series,parcel_path='path/to/brain/atlas/atlas.nii')

~make graph from matrix

graph = matrix_to_igraph(matrix,cost=0.1)

~get community detection results

community_object = graph.community_infomap(edge_weights='weight')
community_array = community.membership

~calculate PC and WMD, return nice brain_graph object with the graph and community object attached
brain_graph = brain_graph(community_object)

Example 2:

~do all over the above, but recursively find communities from all costs (a la Power et al, 2011)
brain_graph = recrusive_network_partition(subject_path='path/to/where/epi/files/live/*',graph_cost=.1,min_community_size=10)
