""" Various helper functions, mostly having to do with data. """
import os
import pandas
import numpy as np

import rl
import roi

# TODO need mean coaster param function

# TODO TEST ALL
def sub_map(num):
    """ Return the directory name, given a subject <num>ber, 
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


def get_trials():
    """ Return trials in trial-time (equivilant to 3 TRs). """
    
    return [1,1,4,4,6,0,4,2,4,1,0,0,0,0,5,4,5,1,5,5,3,6,6,0,0,2,2,3,
            3,1,3,2,4,2,2,6,0,5,3,1,2,2,0,4,3,0,0,6,5,6,6,5,1,0,0,
            5,5,2,2,6,2,0,0,0,6,6,5,0,0,0,0,4,6,4,5,6,4,0,0,0,0,3,3,
            4,2,5,5,1,0,3,3,1,1,0,6,1,5,3,3,0,4,6,0,0,0,1,2,2,0,5,0,
            4,4,4,3,3,2,0,1,0,0,0,4,4,0,0,2,1,6,6,2,4,4,1,1,1,5,4,0,6,
            0,3,3,5,5,4,2,1,1,1,6,1,1,0,2,0,0,6,5,6,0,3,4,3,3,6,4,0,6,6,
            6,1,1,3,6,3,0,5,5,3,0,0,0,2,2,2,3,2,6,3,5,5,0,1,0,2,5,2,4,1,
            4,4,4,0,5,5,6,0,3,4,0,5,0,0,6,1,3,5,0,3,0,0,1,6,3,0,2,2,0,5,
            5,2,5,0,1,4,1,2,0,0,3,0,0,1,3,2,1,4,0,3,5,0,0,4,0,5,2,6,
            6,2,1,6,0,2,3,3,6,4,4,2]


def get_behave_data(num):
    """ In a dict, get and return the behavioral data for subject 
    <num>ber. """
    
    # Use pandas Dataframe to
    # isolate <num> data, then...
    table = pandas.read_table(
            os.path.join(roi.__path__[0], 'exps','catreward',
                    '101-118_regression_NA_AsZero.txt'))
    stable = table[table['sub'] == num]

    # convert to a dict.
    data = {}
    [data.update({colname:series.tolist()}) for 
            colname, series in stable.iteritems()]

    return data

