""" A module process trial timing information. """
import numpy as np


def dtime(trials, duration, drop=None):
    """ Map <trials>, a sequences of values (could be integers,
    representing trial conditions or real valued data from model) 
    into another temporal space, using duration, an integer, 
    specifiying the length of each trial.
    
    <drop> allows you to create sub-trial/duration time 
    representations. It is a binary list of length duration.  
    '1' mean drop that entry, 0 means keep.  Dropped trial
    codes are also 0.
    
    For example, if trial was 2 and duration was 3
    the new represenation would be for trial would be 
    [2, 2, 2] if drop was None or [0, 0, 0].  
    
    However if drop was [0, 1, 0] the trial would become
    [2, 0, 2].  Likwise if drop was [1, 0, 1] trial would be
    [0, 2, 0].
        
    Returns a list of the mapped trials. """

    trials = np.array(trials)
        
    if (trials.ndim > 1):
        raise TypeError("trials must be 1d.")

    if not isinstance(duration, int):
        raise ValueError("duration must be an integer.")

    dtrials = []
    if drop == None:    
        [dtrials.extend([trial, ] * duration) for trial in trials]
    else:
        mask = np.array(drop) == 1
        for trial in trials:
            dtrial = np.array([trial, ] * duration)
            dtrial[mask] = 0
            dtrials.extend(dtrial.tolist())

    return dtrials


# TODO debug, test
def add_empty(data, trials):
    """ Collected behavoiral data may not include (empty)
    jitter periods.  This function corrects for that, adding 
    zero-filled rows to data when conditions is zero. 
    
    <conditions> shouls be an integer sequence of trial events.
        '0' indicates a jitter period.  In implictly assumes 
        jitter and trial lengths are the same.
    <data> should be a 1 or 2 d array-like object. """
    
    conditions = np.array(conditions, dtype=np.uint32)
        ## Cast as int, if it fails, good. 
        ## We only want int.

    data = np.array(data)
    if data.ndim > 2:
        raise ValueError("data must be 1 or 2 d.")

    # Init, assuming 2d, 
    # dropping back to 1d.
    mappedata = np.array([])
    try:
        mappedata = np.zeros((conditions.shape[0], data.shape[1]))
    except IndexError:
        mappedata = np.zeros(conditions.shape[0])
    
    # map: data -> cdata, if conditions != 0
    mappedata[conditions != 0] = data[:,]
        ## This will die if the number of rows in data
        ## does not match the number of non-zero conditions,
        ## which is what we want.
        
    return mappedata
    
