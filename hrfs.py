"""A collection of hemodynamic response functions. """
import numpy as np
import nitime as nt


def double_gamma(self, width=32, a1=6.0, a2=12.0, b1=0.9, b2=0.9, c=0.35):
    """ Returns a HRF.  Defaults are the canonical parameters. """

    x_range = np.arange(0, width, self.TR)    
    d1 = a1 * b1
    d2 = a2 * b2

    # Vectorized...
    hrf = ((x_range / d1) ** a1 * np.exp((d1 - x_range) / b1)) - (
            c * (x_range / d2) ** a2 *np.exp((d2 - x_range) / b2))

    return hrf
    

def mean_fir(self, window_size=30):
    """ Estimate and return the average (for all condtions in trials) 
    finite impulse-response model using self.bold and self.trials. 
    
    <window_size> is the expected length of the haemodynamic response
    in TRs. """

    if self.bold == None:
        raise ValueError(
                'No bold signal is defined. Try create_bold(...)?')
 
    # Convert  self.bold (an array) to a nitime TimeSeries
    # instance
    ts_bold = nt.TimeSeries(
            self.bold, sampling_interval=self.TR)

    # And another one for the events (the different stimuli):
    ts_trials = nt.TimeSeries(
            self.trials, sampling_interval=self.TR)

    # Create a nitime Analyzer instance.
    eva = nt.analysis.EventRelatedAnalyzer(
            ts_bold, ts_trials, window_size)

    # Now do the find the event-relared averaged by FIR:
    # For details see the nitime module and,
    #
    # M.A. Burock and A.M.Dale (2000). Estimation and Detection of 
    # Event-Related fMRI Signals with Temporally Correlated Noise: A 
    # Statistically Efficient and Unbiased Approach. Human Brain 
    # Mapping, 11:249-260
    hrf = eva.FIR.data
    if hrf.ndim == 2: 
        hrf = hrf.mean(0)
        ## hrf if the mean of all 
        ## conditions in trials

    return hrf

