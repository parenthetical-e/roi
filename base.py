""" Class templates for parametric fMRI ROI analyses, 
done programatically. """
import os
import re
import nibabel
import nitime
import roi

import numpy as np

from copy import deepcopy
try:
    from scikits.statsmodels.api import GLS
except ImportError:
    from statsmodels.api import GLS


class Roi():
    """
    The basic. It does OLS regression. 
    """
    
    def __init__(self, TR, roi_name, trials, durations, data):
        # ---
        # User defined variables
        self.TR = TR
        self.roi_name = roi_name
        self.trials = np.array(trials)
        self.durations = np.array(durations)

        self.data = data
        if self.data == None:
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
        Use extract_results() to store the simulation's state.
        
        This private method just extracts relevant data from the regression
        model into a dict.
        """
        
        tosave = {
            'beta':'params',
            't':'tvalues',
            'f_pvalue':'f_pvalue',
            'fvalue':'fvalue',
            'p':'pvalues',
            'r':'rsquared',
            'r_adj':'rsquared_adj',
            'ci':'conf_int',
            'resid':'resid',
            'wresid':'wresid',
            'aic':'aic',
            'bic':'bic',
            'llf':'llf',
            'bse':'bse',
            'nobs':'nobs',
            'df_model':'df_model',
            'df_resid':'df_resid',
            'ess':'ess',
            'ssr':'ssr',
            'mse_model':'mse_model',
            'mse_resid':'mse_resid',
            'mse_total':'mse_total',
            'pretty_summary':'summary'}
        
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

        # Then use nitime to 
        # high pass filte using FIR
        # (~1/128 s, same cutoff as SPM8's default)

        # FIR did well in:
        #
        # Comparison of Filtering Methods for fMRI Datasets
        # F. Kruggela, D.Y. von Cramona, X. Descombesa
        # NeuroImage 10 (5), 1999, 530 - 543.
        filtered = np.ones_like(arr)
        try:
            # Try 2d first...
            for col in range(arr.shape[1]):         
                tsi = nitime.TimeSeries(arr[:,col], 1, self.TR)
                fsi = nitime.analysis.FilterAnalyzer(tsi, ub=None, lb=0.008)
                filtered[...,col] = fsi.fir.data
        except IndexError:
            # Fall back to 1d
            tsi = nitime.TimeSeries(arr, 1, self.TR)
            fsi = nitime.analysis.FilterAnalyzer(tsi, ub=None, lb=0.008)
            filtered = fsi.fir.data

        return filtered


    def _normalize_array(self, arr, function_name):
        """ Normalize the <arr>ay using the <function_name> 
        of one of the functions in roi.norm """

        return getattr(roi.norm, function_name)(arr)


    def _orth_dm(self):
        """ Orthgonalize (by regression) each col in self.dm with respect to 
        its left neighbor. """
        
        dm = self.dm  
            ## Rename for brevity
        
        # Make sure conds and ncol dm
        # are divisors
        conds = list(set(self.trials))
        nconds = len(conds) - 1;     ## Drop baseline
        ncols = dm.shape[1] - 1
        if ncols % nconds:
            raise ValueError(
                "The number of condtions and shape of the dm are incompatible.")

        # If these are the same size there is nothing to
        # orthgonalize.
        if ncols != nconds:
            orth_dm = np.zeros_like(dm)
            orth_dm[:,0] = dm[:,0]
                ## Move baseline data over

            # Use num_col_per_cond, along with nconds 
            # to find the strides we need to take along
            # the DM to orthgonalize each set of col(s) 
            # belonging to each cond.
            num_col_per_cond = ncols / nconds
            for cond in conds:
                # Skip baseline
                if cond == 0: continue
                left = cond
                right = cond + nconds
                
                # Rolling loop over the cols_per_cond
                # orthgonalizing as we go.
                for cnt in range(num_col_per_cond-1):
                    # Orthgonalize left col to right....
                    glm = GLS( dm[:,right], dm[:,left]).fit()  ## GLS(y, x)
                    orth_dm[:,right] = glm.resid
                    orth_dm[:,left] = dm[:,left]
                   
                    # Shift indices for next iteration.
                    left = deepcopy(right)
                    right = right + nconds

            self.dm = orth_dm
        else:
            print("Nothing to orthgonalize.")


    def _write_bold(self):
        """ Write the bold signal to a file in ./bold. """
        
        no_exten = re.split('\.', self.roi_name)[0]
        pathsplit = re.split('/', no_exten)
        drop_path = pathsplit[-1]
        name = drop_path + '_bold.txt'
            ## Grab the non-file extension, non-path, 
            ## parts of roi_name and
            ## append a new extention.
        
        # Create a bold dir nested where roi_name is stored, 
        # if needed that is.
        bold_path = '/'.join(pathsplit[0:-1] + ['bold', ])
        if not os.path.exists(bold_path):
            os.mkdir(bold_path)

        # Write!
        fid = open('/'.join([bold_path, name]), 'w')
        self.bold.tofile(fid, sep='\n')
        fid.close()


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


    def create_dm(self, drop=None, convolve=True):
        """ Create a unit (boxcar-only) DM with one columns for each 
        condition in self.trials.  
        
         If <convolve> the dm is convolved with the HRF (self.hrf). """

        cond_levels = sorted(list(set(self.trials)))
            ## Find and sort conditions in trials

        # Some useful counts...
        num_conds = len(cond_levels)
        num_tr = np.sum(self.durations)
        num_trials = len(self.trials)

        # Map each condition in trials to a
        # 2d binary 2d array.  Each row is a trial
        # and each column is a condition.
        dm_unit = np.zeros((num_tr, num_conds))
        for col, cond in enumerate(cond_levels):
            # Create boolean array use it to 
            # populate the dm with ones... 
            # which must be in tr time.
            mask_in_tr = roi.timing.dtime(
                    self.trials == cond, self.durations, None, False)

            dm_unit[mask_in_tr,col] = 1

        self.dm = dm_unit
        
        if convolve:
            self.dm = self._convolve_hrf(self.dm)


    def create_dm_param(self, names, drop=None, box=True, 
            orth=False, convolve=True):
        """ Create a parametric design matrix based on <names> in self.data. 
        
        If <box> a univariate dm is created that fills the leftmost
        side of the dm.

        If <orth> each regressor is orthgonalized with respect to its
        left-hand neighbor (excluding the baseline).
        
        If <convolve> the dm is convolved with the HRF (self.hrf). """

        cond_levels = sorted(list(set(self.trials)))
            ## Find and sort conditions in trials
       
        # Some useful counts...
        num_trials = len(self.trials)
        num_names = len(names)
        num_tr = np.sum(self.durations)
        
        dm_param = None
            ## Will eventually hold the 
            ## parametric DM.

        for cond in cond_levels:
            if cond == 0:
                continue
                    ## We add the baseline 
                    ## in at the end

            # Create a temp dm to hold this
            # condition's data
            dm_temp = np.zeros((num_tr, num_names))

            mask_in_tr = roi.timing.dtime(
                    self.trials == cond, self.durations, drop, False)

            # Get the named data, convert to tr time 
            # then add to the temp dm using the mask
            for col, name in enumerate(names):
                data_in_tr = roi.timing.dtime(
                            self.data[name], self.durations, drop, 0)

                dm_temp[mask_in_tr,col] = data_in_tr[mask_in_tr]
                    
            # Store the temporary DM in 
            # the final DM.
            if dm_param == None:
                dm_param = dm_temp  ## reinit
            else:
                dm_param = np.hstack((dm_param, dm_temp))  ## adding

        # Create the unit DM too, then combine them.
        # defining self.dm in the process
        self.create_dm(convolve=False)
        dm_unit = self.dm.copy(); self.dm = None  
            ## Copy and reset

        if box:
            self.dm = np.hstack((dm_unit, dm_param))
        else:
            baseline = dm_unit[:,0]
            baseline = baseline.reshape(baseline.shape[0], 1)
            self.dm = np.hstack((baseline, dm_param))
                ## If not including the boxcar,
                ## we still need the baseline model.

        # Orthgonalize the regessors?
        if orth: 
            self._orth_dm()

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
        mdata = np.ma.MaskedArray(data=data, mask=mask)

        # Use masked array to average over x, y, z
        # only for non-zero fMRI data, 
        # resulting in a 1d times series
        self.bold = np.asarray(mdata.mean(0).mean(0).mean(0).data)
        
        # Now archive the bold signal, 
        # prior to any preprocessing
        self._write_bold()

        if preprocess:
            self.bold = self._filter_array(self.bold)
        

    def fit(self, norm='zscore'):
        """ Calculate the regression parameters and statistics. """
        
        bold = self.bold.copy()
        dm = self.dm.copy()
        
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
        
        # Truncate bold or dm_dummy if needed, and Go!
        try:
            bold = bold[0:dm_dummy.shape[0]]
            dm_dummy = dm_dummy[0:len(bold),:]
        except IndexError:
            pass

        self.glm = GLS(bold, dm_dummy).fit()
    

    def contrast(self, contrast, name):
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

        self.create_dm(convolve=True)

        self.fit(norm='zscore')


    def print_model_summary(self):
        """ Prints all defined model names and their docstrings. """

        # find all self.model_N attritubes and run them.
        all_attr = dir(self)
        all_attr.sort()
        past_models = []
        model_count = 0
        for attr in all_attr:
            a_s = re.split('_', attr)
            
            # Match only model_N where N is an integer
            if len(a_s) == 2:
                if (a_s[0] == 'model') and (re.match('\A\d+\Z', a_s[1])):
                    
                    model_count += 1
                    model = attr
                        ## Rename for clarity

                    # Model name must be unique.
                    if model in past_models:
                        raise AttributeError(
                                '{0} was not unique.'.format(model))
                    past_models.append(model)

                    # Now call the model and
                    # print its info out.
                    print("{0}. {1}:".format(model_count, model))
                    try:
                        func = getattr(self, model)
                        print(func.im_func.func_doc)                    
                    except KeyError:
                        print("Data not Found.  Moving on.")
                        continue

    
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
                    
                    model = attr
                        ## Rename for clarity

                    # Model name must be unique.
                    if model in past_models:
                        raise AttributeError(
                                '{0} was not unique.'.format(model))
                    past_models.append(model)

                    # Now call the model and
                    # save its results.
                    print('Fitting {0}.'.format(model))
                    try:
                        getattr(self, model)()
                    except KeyError:
                        print("Data not Found.  Moving on.")
                        continue

                    self.extract_results(name=model)
        
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
            'duration;fs':'durations',
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


class Mean(Roi):
    """ A variant of Roi that uses averaged BOLD data, from
    a text file instaed of a nii. 
    
    <roi_name> should be the path to that file. """


    def __init__(self, TR, roi_name, trials, durations, data):
        Roi.__init__(self, TR, roi_name, trials, durations, data)   

    
    def create_bold(self, preprocess=True):
        """ Read in the fMRI data from <roi_name> and <preprocess> it,
        if True. 
        
        The BOLD file should contain in a single column, like that 
        made by _write_bold() """
        
        self.bold = np.loadtxt(self.roi_name)
        
        if preprocess:
            self.bold = self._filter_array(self.bold)

