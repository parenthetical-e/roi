""" An (master) roi experiment for the catreward project. """
# TODO - test all!

import numpy as np
import rl
import roi
from roi.base import Roi

def sub_map(num):
	""" Return the directory name, given a subject <num>be, 
	e.g. 101. """
	
	# Create the map,
	# and use it
	num_dir_map = {
	101:'101M80351917',
    102:'102M80359344',
	103:'103M80358136',
	104:'104M80368842',
	105:'105M80350861',
	106:'106M80381623',
	108:'108M80357327',
	109:'109M80328568',
	111:'111M80343408',
	112:'112M80364602',
	113:'113M80380288',
	114:'114M80371580',
	115:'115M80364638',
	116:'116M80363397',
	117:'117M80354305',
	118:'118M80330684'}
	
	return num_dir_map[num]


def _trials(use, drop=None):
	""" Return trials in TR time, use 'condition' or 'combined'. """
	
	trials = [1,1,4,4,6,0,4,2,4,1,0,0,0,0,5,4,5,1,5,5,3,6,6,0,0,2,2,3,
			3,1,3,2,4,2,2,6,0,5,3,1,2,2,0,4,3,0,0,6,5,6,6,5,1,0,0,
			5,5,2,2,6,2,0,0,0,6,6,5,0,0,0,0,4,6,4,5,6,4,0,0,0,0,3,3,
			4,2,5,5,1,0,3,3,1,1,0,6,1,5,3,3,0,4,6,0,0,0,1,2,2,0,5,0,
			4,4,4,3,3,2,0,1,0,0,0,4,4,0,0,2,1,6,6,2,4,4,1,1,1,5,4,0,6,
			0,3,3,5,5,4,2,1,1,1,6,1,1,0,2,0,0,6,5,6,0,3,4,3,3,6,4,0,6,6,
			6,1,1,3,6,3,0,5,5,3,0,0,0,2,2,2,3,2,6,3,5,5,0,1,0,2,5,2,4,1,
			4,4,4,0,5,5,6,0,3,4,0,5,0,0,6,1,3,5,0,3,0,0,1,6,3,0,2,2,0,5,
			5,2,5,0,1,4,1,2,0,0,3,0,0,1,3,2,1,4,0,3,5,0,0,4,0,5,2,6,
			6,2,1,6,0,2,3,3,6,4,4,2]

	dtrials = roi.timing.dtime(trials, duration=3, drop=drop)	
	if use == 'condition':
		return dtrials
	elif use == 'combined':
		dtrials_recode = np.array(dtrials)
		dtrials_recode[dtrials_recode > 0] = 1
		return dtrials_recode.tolist()
	else:
		raise ValueError("use not understood.  Try: condition or combined")                                         


def create_data(num):
	""" Reads or create all the data needed to do an analysis
	for subject <num>ber. """
	# TODO
	pass
	
	
def run(num):
	""" Run a catreward experiment for subject <num>ber."""
	# TODO
	roi_names = []
	
	pass
	
class Catreward(Roi):
	""" A Roi analysis class, customized for the catreward project. """

	def __init__(self):
		super(Catreward, self).__init__()
	
		self.create_bold(preprocess=True)
        self.create_hrf(
			function_name='mean_fir', params={'window_size':24})
		
       self.data['meta']['bold'] = self.roi_name


	def model_01(self):
        """ Model each stimulus as seperate regressor. """

		data_to_use = ['conds']	
        self.data['meta']['bold'] = self.roi_name
        self.data['meta']['dm'] = data_to_use
        
		self.create_dm(names=data_to_use, convolve=True)
        self.fit(norm='zscore')


	def model_02(self):
        """ Model reponses as seperate regressors. """

		# TODO this is going to need custom work... to work.
		# TODO Do not use yet!
		data_to_use = ['resps']
        self.data['meta']['dm'] = data_to_use

        self.create_dm(names=data_to_use, convolve=True)
        self.fit(norm='zscore')	


	def model_03(self):
        """ Model reaction times. """

		data_to_use = ['rt']		
        self.data['meta']['dm'] = data_to_use

        self.create_dm(names=data_to_use, convolve=True)
        self.fit(norm='zscore')

	
	def model_040(self):
        """ Model similarity, for each category. """

		data_to_use = ['distance']
        self.data['meta']['dm'] = data_to_use

        self.create_dm(names=data_to_use, convolve=True)
        self.fit(norm='zscore')
		
	
	def model_041(self):
        """ Model similarity using both categories. """

		data_to_use = ['distance_both']
        self.data['meta']['dm'] = data_to_use

        self.create_dm(names=data_to_use, convolve=True)
        self.fit(norm='zscore')


	def model_05(self):
        """ Model gains and losses in a single regressor. """

		data_to_use = ['gl']
        self.data['meta']['dm'] = data_to_use

        self.create_dm(names=data_to_use, convolve=True)
        self.fit(norm='zscore')	


	def model_06(self):
        """ Model behavioral accuracy. """

		data_to_use = ['acc']
        self.data['meta']['dm'] = data_to_use

        self.create_dm(names=data_to_use, convolve=True)
        self.fit(norm='zscore')


	def model_07(self):
        """ Model RPE. """

		data_to_use = ['rpe']
        self.data['meta']['dm'] = data_to_use

        self.create_dm(names=data_to_use, convolve=True)
        self.fit(norm='zscore')


	def model_08(self):
        """ Model value. """

		data_to_use = ['value']
        self.data['meta']['dm'] = data_to_use

        self.create_dm(names=data_to_use, convolve=True)
        self.fit(norm='zscore')


	def model_09(self):
        """ Model accuracy, diminishing it by similarity 
		for each category (separately). """

		data_to_use = ['acc_d']
        self.data['meta']['dm'] = data_to_use

        self.create_dm(names=data_to_use, convolve=True)
        self.fit(norm='zscore')


	def model_10(self):
        """ Model RPE, diminishing it by similarity 
		for each category (separately). """

		data_to_use = ['rpe_d']
        self.data['meta']['dm'] = data_to_use

        self.create_dm(names=data_to_use, convolve=True)
        self.fit(norm='zscore')


	def model_11(self):
        """ Model value, diminishing it by similarity 
		for each category (separately). """

		data_to_use = ['value_d']
        self.data['meta']['dm'] = data_to_use

        self.create_dm(names=data_to_use, convolve=True)
        self.fit(norm='zscore')


	def model_12(self):
        """ Model accuracy, diminishing it by similarity 
		difference between the two categories. """

		data_to_use = ['acc_d_diff']
        self.data['meta']['dm'] = data_to_use

        self.create_dm(names=data_to_use, convolve=True)
        self.fit(norm='zscore')


	def model_13(self):
        """ Model RPE, diminishing it by similarity 
		difference between the two categories. """

		data_to_use = ['rpe_d_diff']
        self.data['meta']['dm'] = data_to_use

        self.create_dm(names=data_to_use, convolve=True)
        self.fit(norm='zscore')


	def model_14(self):
        """ Model value, diminishing it by similarity 
		difference between the two categories. """

		data_to_use = ['value_d_diff']
        self.data['meta']['dm'] = data_to_use

        self.create_dm(names=data_to_use, convolve=True)
        self.fit(norm='zscore')

