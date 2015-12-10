import os, sys, math
import numpy as np
import nibabel as nib

# can be called for any task
def get_taskstr(task):
	return 'tfMRI_{task}_RL'.format(task=task)

def get_evdir(sub, task):
	'''Return the directory which contains the FSL 3-col onset ('EV') files for a specified subject and task. Implementation logic goes here.'''
	task_str = get_taskstr(task)
	metadata_dir = '/home/despoB/connectome-raw/%s/MNINonLinear/Results'%sub
	return os.path.join(metadata_dir, task_str, 'EVs')

def get_epidir(sub, task):
	'''Return the directory which contains the epi files for a specified subject and task. Implementation logic goes here.'''
	task_str = get_taskstr(task)
	data_dir = '/home/despoB/connectome-data/%s'%sub
	return os.path.join(data_dir, task_str)

def get_epifile(sub, task):
	'''Return the directory which contains the epi files for a specified subject and task. Implementation logic goes here.'''
	task_str = get_taskstr(task)
	epi_dir = get_epidir(sub, task)
	return os.path.join(epi_dir, '%s.nii.gz'%task_str)

def get_cond_timeseries(sub, task, conds):
	'''Reads the onset/EV files for the specified subject and task, and splits them according to a dict of conditions.

	sub	:Subject number, as a string
	task	:Task names
	conds	:dict like {'condname':['file1.txt','file2.txt'], 'cond2':['file3.txt']}
	'''
	ev_dir = get_evdir(sub, task)
	evfile_list = set(os.listdir(ev_dir))
	
	epi_fn = get_epifile(sub, task)
	# load the file, get the TR
	epi = nib.load(epi_fn)
	print(epi_fn, epi, epi.shape)

	TR = 0.72 # seconds; figure out where to embed or obtain this knowledge. should be correct for HCP.

	cond_timeseries = dict()

	# check that all the filenames given in conds exist in the dir
	for condition in conds.keys():
		cond_evfiles = set(conds[condition]) # eliminates duplicates anyway
		if not evfile_list.issuperset(cond_evfiles):
			print('Error! Missing EV files for condition {c}, make sure {f} exist'.format(c=condition,f=cond_evfiles))
			return None # or throw an Error

		# all files exist, since we didn't return or throw an error
		cond_tps = set()
		for evfile in cond_evfiles:
			f = open(os.path.join(ev_dir,evfile),'r')
			for line in f:
				# handle each event listing. these are all allowed to overlap with each other (long events relative to ITI)
				# the format is onset	duration	amplitude, we only care about the first two
				# eventually we just want the set of timepoints of the events and no others
				[onset, duration, amp] = [float(p) for p in line.split()]
				first_idx = int(math.floor(onset/TR)) # not using ceil here because if an event began at TR 10.5, we want vol 11 onwards, which has index 10 as python is 0-based
				duration_TR = int(math.ceil(duration/TR))
				last_idx = first_idx + duration_TR # inclusive! we want this index as well
				print (onset, onset/TR, duration, duration_TR, first_idx, last_idx)
				idxs = range(first_idx, last_idx+1)
				assert(min(idxs)==first_idx and max(idxs)==last_idx)
				cond_tps.update(idxs)

		cond_timeseries[condition] = cond_tps
	else: # will be executed when the 'for condition in conds.keys()' loop terminates by exhaustion, ie all conditions are checked and no errors were thrown
		# sanity checks
		# make sure no timepoints are shared with other existing conditions! this is a hard constraint.
		assert(len(set.intersection(*cond_timeseries.values()))==0)
		# any further checks go here
		# return the sorted and labeled timeseries
		cond_timeseries_sorted = {condition: sorted(timepoints) for condition, timepoints in cond_timeseries.items()}
		return cond_timeseries_sorted


tasks = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']

for task in tasks:
	# per-subj setup
	sub = '987983'

	# per-task setup; kinda unpythonic, but allows explicit setup of conditions
	conds = dict()
	if task == 'EMOTION':
		conds['fear'] = ['fear.txt'] # names of EV files that should be grouped into this condition
		conds['neut'] = ['neut.txt']
		cond_ts = get_cond_timeseries(sub, task, conds)
		print cond_ts



