""" An (master) roi experiment for the catreward project. """
from roi.base import Roi


class Catreward(Roi):
    """ A Roi analysis class, customized for the catreward project. """

    def __init__(self):
        Roi.__init__(self)
    
        self.create_bold(preprocess=True)
        self.create_hrf(
            function_name='mean_fir', params={'window_size':24})
        
        self.data['meta']['bold'] = self.roi_name
 

    def model_020(self):
        """ Model responses as separate regressors. """

        data_to_use = ['resp1', 'resp6']
        self.data['meta']['dm'] = data_to_use

        self.create_dm_param(names=data_to_use, box=False)
        self.fit(norm='zscore') 


    def model_021(self):
        """ Model the correct responses as separate regressors. """

        data_to_use = ['cresp1', 'cresp6']
        self.data['meta']['dm'] = data_to_use

        self.create_dm_param(names=data_to_use, box=False)
        self.fit(norm='zscore')


    def model_03(self):
        """ Model reaction times. """

        data_to_use = ['rt']        
        self.data['meta']['dm'] = data_to_use

        self.create_dm_param(names=data_to_use)
        self.fit(norm='zscore')

    
    def model_040(self):
        """ Model similarity, for each category. """

        data_to_use = ['distance']
        self.data['meta']['dm'] = data_to_use

        self.create_dm_param(names=data_to_use)
        self.fit(norm='zscore')
        
    
    def model_041(self):
        """ Model similarity using both categories. """

        data_to_use = ['distance_both']
        self.data['meta']['dm'] = data_to_use

        self.create_dm_param(names=data_to_use)
        self.fit(norm='zscore')


    def model_05(self):
        """ Model gains and losses in a single regressor. """

        data_to_use = ['gl']
        self.data['meta']['dm'] = data_to_use

        self.create_dm_param(names=data_to_use)
        self.fit(norm='zscore') 


    def model_06(self):
        """ Model behavioral accuracy. """

        data_to_use = ['acc']
        self.data['meta']['dm'] = data_to_use

        self.create_dm_param(names=data_to_use)
        self.fit(norm='zscore')


    def model_07(self):
        """ Model RPE. """

        data_to_use = ['rpe']
        self.data['meta']['dm'] = data_to_use

        self.create_dm_param(names=data_to_use)
        self.fit(norm='zscore')


    def model_08(self):
        """ Model value. """

        data_to_use = ['value']
        self.data['meta']['dm'] = data_to_use

        self.create_dm_param(names=data_to_use)
        self.fit(norm='zscore')


    def model_09(self):
        """ Model accuracy, diminishing it by similarity 
        for each category (separately). """

        data_to_use = ['acc_d']
        self.data['meta']['dm'] = data_to_use

        self.create_dm_param(names=data_to_use)
        self.fit(norm='zscore')


    def model_10(self):
        """ Model RPE, diminishing it by similarity 
        for each category (separately). """

        data_to_use = ['rpe_d']
        self.data['meta']['dm'] = data_to_use

        self.create_dm_param(names=data_to_use)
        self.fit(norm='zscore')


    def model_11(self):
        """ Model value, diminishing it by similarity 
        for each category (separately). """

        data_to_use = ['value_d']
        self.data['meta']['dm'] = data_to_use

        self.create_dm_param(names=data_to_use)
        self.fit(norm='zscore')


    def model_12(self):
        """ Model accuracy, diminishing it by similarity 
        difference between the two categories. """

        data_to_use = ['acc_d_diff']
        self.data['meta']['dm'] = data_to_use

        self.create_dm_param(names=data_to_use)
        self.fit(norm='zscore')


    def model_13(self):
        """ Model RPE, diminishing it by similarity 
        difference between the two categories. """

        data_to_use = ['rpe_d_diff']
        self.data['meta']['dm'] = data_to_use

        self.create_dm_param(names=data_to_use)
        self.fit(norm='zscore')


    def model_14(self):
        """ Model value, diminishing it by similarity 
        difference between the two categories. """

        data_to_use = ['value_d_diff']
        self.data['meta']['dm'] = data_to_use

        self.create_dm_param(names=data_to_use)
        self.fit(norm='zscore')

