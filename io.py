""" Functions for reading and writing of Exp results """
import csv
import h5py
import numpy as np
import nibabel as nb


def _walk(dictionary, hdf):
    """ 
    Recursively walk the provided dict <d> creating groups or saving data 
    in <hdf>, as appropriate.
    """
    for key, val in dictionary.items():
        if isinstance(val, dict):
            hdfnext = hdf.create_group(key)
            _walk(val, hdfnext)
                ## gettin' all recursive and shit, yo.
        else:
            if val is None: 
                val = 0
                ## h5py does not know 
                ## what to do with None.
            
            data = np.array(val)
            hdf.create_dataset(key, data=data)


def write_hdf(results, name):
    """ 
    Iterate over the <results> list, mimicking the hierarchical structure
    of each entry.  Name the resulting file <name>.
    """
    from simfMRI.io import _walk
    
    hdf = h5py.File(name, 'w')
    for ii, res in enumerate(results):
        # Create a top level group for each res
        # in results.  Then recursively walk res.
        # Anything that is not a dict is 
        # assumed to be data.
        hdf_ii = hdf.create_group(str(ii))
        _walk(res, hdf_ii)
    
    hdf.close()


def write_nifti(nifti, name):
    """ Write out the <nifti> object, name it <name>. """
    
    nb.save(nifti, name)


def read_hdf(name, path='/model_01/bic'):
    """ 
    In the <hdf> file, for every top-level node return the 
    data specified by path.
    """

    # First locate the dataset in the first result,
    # then grab that data for every run.
    hdf = h5py.File(name, 'r')

    return [hdf[node + path].value for node in hdf.keys()]


def read_hdf_inc(name, path='/model_01/bic'):
    """ 
    In <name> (a hdf file), for every top-level node *incrementally* 
    return the data specified by <path>.
    """

    # First locate the dataset in the first result,
    # then grab that data for every run.
    hdf = h5py.File(name, 'r')

    for node in hdf.keys():
        yield hdf[node + path].value


def read_nifti(name):
    """ A simple by <name> nifti loader, returns a nifti object. """

    return nb.nifti1.load(name)


def read_trials(name):
    """ Read in the trials from <name>, returning a list. """
    
    # Open a file handle to <name>
    # slurp it up with csv, and put
    # it in a list.  Returning that
    # list.
    fhandle = open(name, 'r')
    trials_f = csv.reader(fhandle).next()
    trials = [int(trl) for trl in trials_f]
    fhandle.close()

    return trials

  
def read_roi_csv(name):
    """ Read in <name>ed roi data. """

    return np.loadtxt(name, delimiter=',')


def get_results_meta(name, model):
    """ Get the BOLD and DM metadata for <model> from the hdf file
    <name>. """

    hdf = h5py.File(name,'r')

    meta = {}
    meta['bold'] = hdf['/0/' + model + '/data/meta/bold'].value
    meta['dm'] = hdf['/0/'+ model + '/data/meta/dm'].value.tolist()

    return meta

