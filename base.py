""" Class templates for parametric fMRI ROI analyses, 
done programatically. """
import re
import nibabel
import nitime
import roi

import numpy as np

from copy import deepcopy
from scikits.statsmodels.api import GLS


class Roi():
    """
    The basic. It does OLS regression. 
    """
    
    def __init__(self, TR, roi_name, trials, data):
        # ---
        # User defined variables
        self.TR = TR
        self.trials = trials
        self.roi_name = roi_name

        if data == None:
            self.data = {}

        self.data['meta'] = {}  ## meta is for model metadata

        # ---
        # Intialize model data structues,
        # these take on values by calling:
        self.hrf = None     ## self.create_hrf()
        self.dm = None      ## self.create_dm() or create_dm_pararm()
        self.bold = None    ## self.create_bold()

        # --- 
        # The two flavors of results
        self.glm = None     ## self.fit(), fit a regrssion model
        self.results = {}   ## self.extract_results() into a dict


    def _convolve_hrf(self, arr):
        """
        Convolves hrf basis with a 1 or 2d (column-oriented) array.
        """
        
        # self.hrf may or may not exist yet
        if self.hrf == None:
            raise ValueError('No hrf is defined. Try self.create_hrf()?')
        
        arr = np.asarray(arr)   ## Just in case 

        # Assume 2d (or really N > 1 d), 
        # fall back to 1d.
        arr_c = np.zeros_like(arr)
        try:
            for col in range(arr.shape[1]):
                arr_c[:,col] = np.convolve(
                        arr[:,col], self.hrf)[0:arr.shape[0]]
                    ## Convolve and truncate to length
                    ## of arr
        except IndexError:
            arr_c = np.convolve(arr[:], self.hrf)[0:arr.shape[0]]
        
        return arr_c


    def _reformat_model(self):
        """
        Use save_state() to store the simulation's state.
        
        This private method just extracts relevant data from the regression
        model into a dict.
        """

        tosave = {
            'beta':'params',
            't':'tvalues',
            'fvalue':'fvalue',
            'p':'pvalues',
            'r':'rsquared',
            'ci':'conf_int',
            'resid':'resid',
            'aic':'aic',
            'bic':'bic',
            'llf':'llf',
            'mse_model':'mse_model',
            'mse_resid':'mse_resid',
            'mse_total':'mse_total',
            'pretty_summary':'summary'
        }
        
        # Try to get each attr (a value in the dict above)
        # first as function (without args) then as a regular
        # attribute.  If both fail, silently move on.
        model_results = {}
        for key, val in tosave.items():
            try:
                model_results[key] = deepcopy(getattr(self.glm, val)())
            except TypeError:
                model_results[key] = deepcopy(getattr(self.glm, val))
            except AttributeError:
                continue
        
        return model_results


    def _filter_array(self, arr):
        """ Filter and smooth the 1 or 2 d <arr>ay. """    
        
        if len(arr.shape) > 2:
            raise ValueError("<arr> must be 1 or 2d.")
        
        try:
            n_col = arr.shape[1]
        except IndexError:
            n_col = 1

        filtered = np.ones_like(arr)
        for col in range(n_col):         
            # Then use nitime to 
            # high pass filte using FIR
            # (~1/128 s, same cutoff as SPM8's default)

            # FIR did well in:
            #
            # Comparison of Filtering Methods for fMRI Datasets
            # F. Kruggela, D.Y. von Cramona, X. Descombesa
            # NeuroImage 10 (5), 1999, 530 - 543.
            tsi = nitime.TimeSeries(arr[...,col], 1, self.TR)
            fsi = nitime.analysis.FilterAnalyzer(tsi, ub=None, lb=0.008)
            
            filtered[...,col] = fsi.fir.data

        return filtered


    def _normalize_array(self, arr, function_name):
        """ Normalize the <arr>ay using the <function_name> 
        of one of the functions in roi.norm """

        return getattr(roi.norm, function_name)(arr)


    def _orth_array(self, arr):
        """ Orthgonalize each col in (2d) <arr> with respect to its 
        left neighbor.  """

        arr = np.array(arr)
        if arr.ndim != 2:
            raise ValueError("arr must be 2d.")

        orth_arr = np.zeros_like(arr)

        # TODO - the orth

        return orth_arr

    def create_hrf(self, function_name, params=None):
        """ Creates an hemodynamic response model using the 
        function named <function_name> in roi.hrfs using
        <params> which should be a dictionary of parameters that 
        function_name will accept when called.
        
        If <params> is None the default parameters are used, if 
        possible. 
        
        hrf function_names are 'mean_fir' and 'double_gamma'. """

        if params == None:
            self.hrf = getattr(roi.hrfs, function_name)(self)
        else:
            self.hrf = getattr(roi.hrfs, function_name)(self, **params)


    def create_dm(self, convolve=True):
        """ Create a unit (boxcar-only) DM with one columns for each 
        condition in self.trials.  
        
         If <convolve> the dm is convolved with the HRF (self.hrf). """

        trials_arr = np.array(self.trials)
            ## Makes masking easy

        cond_levels = sorted(list(set(self.trials)))
            ## Find and sort unique onsets, 
            ## aka condition levels
       
        # Some useful counts...
        num_trials = len(self.trials)
        num_conds = len(cond_levels)

        # Map each condition in trials to a
        # 2d binary 2d array.  Each row is a trial
        # and each column is a condition.
        dm_unit = np.zeros((num_trials, num_conds))
        for col, cond in enumerate(cond_levels):
            mask = trials_arr == cond
                ## as a boolean array

            dm_unit[mask,col] = 1

        self.dm = dm_unit
        
        if convolve:
            self.dm = self._convolve_hrf(self.dm)


    def create_dm_param(self, names, box=True, orth=False, convolve=True):
        """ Create a parametric design matrix based on <names> in self.data. 
        
        If <box> a univariate dm is created that fills the leftmost
        side of the dm.

        If <orth> each regressor is orthgonalized with respect to its
        left-hand neighbor (excluding the baseline).
        
        If <convolve> the dm is convolved with the HRF (self.hrf). """

        trials_arr = np.array(self.trials)
            ## Makes masking easy

        # Uses names to retrieve data from self.data
        name_data = {}
        [name_data.update({name:self.data[name]}) for name in names]

        cond_levels = sorted(list(set(self.trials)))
            ## Find and sort unique trials, i.e. 
            ## conditions
       
        # Some useful counts...
        num_trials = len(self.trials)
        num_names = len(names)

        # For each cond in trials, create a temporary
        # DM, then loop over each name, using a
        # cond mask to select the right data
        dm_name_data = None
            ## Will hold the final parametric DM.

        for cond in cond_levels:
            dm_temp = np.zeros((num_trials, num_names))

            mask = trials_arr == cond

            for col, name in enumerate(names):
                dm_temp[mask,col] = name_data[name][mask]
                    ## Get the named data, then mask it

            # Store the temporary DM in 
            # the final DM.
            if dm_name_data == None:
                dm_name_data = dm_temp
            else:
                dm_name_data = np.hstack((dm_name_data, dm_temp))

        # Create the unit DM too, then combine them.
        # defining self.dm in the process
        dm_unit = self._create_dm_unit()
        if box:
            self.dm = np.hstack((dm_unit, dm_name_data))
        else:
            self.dm = np.hstack((dm_unit[:,0], dm_name_data))
                ## If not including the boxcar,
                ## we still need the baseline model.

        # Orthgonalize the regessors?
        if orth:
            if (dm_unit.ndim > 1) and (dm_unit.shape[1] > 2):
                raise ValueError("orth can't be used if the univariate DM\
                        has more than one non-baseline entry.")

            self.dm = _orth_arr(self.dm)

        # Convolve with self.hrf?
        if convolve:
            self.dm = self._convolve_hrf(self.dm)


    def create_bold(self, preprocess=True):
        """ Extract the fMRI data from roi_name and <preprocess> it,
        if True. """
        
       # TODO - test 
        nifti = nibabel.nifti1.load(self.roi_name)

        # Isolate non-zero timecourses 
        data = nifti.get_data()
        mask = data < 0.1
            ## Bold data will alway be greater than 0.1
            ## but we want an inverted mask...
        
        # Create a masked array...
        mdata = np.ma.MaskArray(data=data, mask=mask)

        # Use masked array to average over x, y, z
        # only for non-zero fMRI data, 
        # resulting in a 1d times series
        self.bold = np.asarray(mdata.mean(0).mean(0).mean(0).data)
        
        if preprocess:
            self.bold = self._filter_array(self.bold)
    

    def fit(self, norm='zscore'):
        """ Calculate the regression parameters and statistics. """
        
        bold = self.bold
        dm = self.dm
        
        # Normalize both the bold and dm
        if norm != None:
            bold = self._normalize_array(bold, norm)
            dm = self._normalize_array(dm, norm)

        # Add movement regressors... if present
        try:
            dm_movement = self.data['movement']
            dm = np.vstack((dm, dm_movement))
        except KeyError:
            pass
        
        # Append a dummy predictor and run the regression
        #
        # Dummy is added at the last minute so it does not
        # interact with normalization or smoothing routines.
        dm_dummy = np.ones((dm.shape[0], dm.shape[1] + 1))
        dm_dummy[0:dm.shape[0], 0:dm.shape[1]] = dm
        
        # Go!
        self.glm = GLS(bold, dm_dummy).fit()
    
    
    def contrast(self, contrast):
        """ Uses the current model to statistically compare predictors 
        (t-test), returning df, t and p values.
        
        <contrast> - a 1d list of [1,0,-1] the same length as the number
            of predictors in the model (sans the dummy, which is added
            silently). """
        
        if self.glm == None:
            raise ValueError("No glm present.  Try self.fit()?")
        
        contrast = self.glm.t_test(contrast)
            ## This a thin wrapper for 
            ## statsmodels contrast() method

        return contrast.df_denom, contrast.tvalue, contrast.pvalue


    def model_00(self):
        """ The simplest model, a univariate analysis of all conditions in 
        trials. """

        self.data['meta']['bold'] = self.roi_name
        self.data['meta']['dm'] = [str(cond) for cond in set(self.trials)]

        self.create_bold(preprocess=True)
        self.create_hrf(function_name='mean_fir', params={'window_size':24})
        self.create_dm(convolve=True)
        
        self.fit(norm='zscore')


    def run(self, code):
        """
        Run all defined models in order, returning their tabulated results.
        
        <code> - the unique batch or run code for this experiment.
        
        Models are any method of the form 'model_N' where N is an
        integer (e.g. model_2, model_1012 or model_666).  
        Models take no arguments (besides self).
        """
        
        self.results['batch_code'] = code
        
        # find all self.model_N attritubes and run them.
        all_attr = dir(self)
        all_attr.sort()
        past_models = []
        for attr in all_attr:
            a_s = re.split('_', attr)
            
            # Match only model_N where N is an integer
            if len(a_s) == 2:
                if (a_s[0] == 'model') and (re.match('\A\d+\Z', a_s[1])):
                    
                    # Model name must be unique.
                    if attr in past_models:
                        raise AttributeError(
                                '{0} was not unique.'.format(attr))
                    past_models.append(attr)

                    # Now call the model and
                    # save its results.
                    print('Fitting {0}.'.format(attr))
                    getattr(self, attr)()
                    self.extract_results(name=attr)
        
        return self.results
        

    def extract_results(self, name):
        """
        Saves most of the state of the current model to results, keyed
        on <name>.  Saves greedily, trading storage space for security 
        and redundancy.
        """

        tosave = {
            'roi_name':'roi_name',
            'TR':'TR',
            'trials':'trials',
            'data':'data',
            'dm':'dm',
            'hrf':'hrf',
            'bold':'bold'
        }
            ## This list is only for attr hung directly off
            ## of self.
        
        # Add a name to results
        self.results[name] = {}
        
        # Try to get each attr (a value in the dict above)
        # first as function (without args) then as a regular
        # attribute.  If both fail, silently move on.
        for key, val in tosave.items():
            try:
                self.results[name][key] = deepcopy(getattr(self, val)())
            except TypeError:
                self.results[name][key] = deepcopy(getattr(self, val))
            except AttributeError:
                continue

        # Now add the reformatted data from the current model,
        # if any.
        self.results[name].update(self._reformat_model())

