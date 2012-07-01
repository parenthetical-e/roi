""" Functions for reading and writing of Exp results """
import csv
import h5py
import numpy as np
import nibabel as nb


def _walkd(dictionary, hdf):
    """ Recursively walk the provided dict <dictionary> creating groups 
    or saving data in <hdf> (a hdf file object), as appropriate. """
    for key, val in dictionary.items():
        if isinstance(val, dict):
            hdfnext = hdf.create_group(key)
            _walkd(val, hdfnext)
                ## gettin' all recursive and shit, yo.
        else:
            if val is None: 
                val = 0
                ## h5py does not know 
                ## what to do with None.
            
            data = np.array(val)
            hdf.create_dataset(key, data=data)


def write_hdf(results, name):
    """ Iterate over the <results> list (a list of dicts containing, mimicking 
    the hierarchical structure of each entry.  Name the resulting file 
    <name>. """

    hdf = h5py.File(name, 'w')
    for ii, res in enumerate(results):
        # Create a top level group for each res
        # in results.  Then recursively walk res.
        # Anything that is not a dict is 
        # assumed to be data.
        hdf_ii = hdf.create_group(str(ii))
        _walkd(res, hdf_ii)
    
    hdf.close()


def write_nifti(nifti, name):
    """ Write out the <nifti> object, name it <name>. """
    
    nb.save(nifti, name)     


def get_hdf_data(name, path):
    """ In the <hdf> file, for every top-level node return the 
    data specified by path. """

    # First locate the dataset in the first result,
    # then grab that data for every run.
    hdf = h5py.File(name, 'r')
    
    return [hdf[node + path].value for node in sorted(hdf.keys())]


def get_hdf_data_inc(name, path):
    """ In <name> (a hdf file), for every top-level node *incrementally* 
    return the data specified by <path>. """

    # First locate the dataset in the first result,
    # then grab that data for every run.
    hdf = h5py.File(name, 'r')

    for node in sorted(hdf.keys()):
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


def get_model_names(hdf5_name):
    """ Return the model names in <hdf5_name>. """
    # Open,
    hdf = h5py.File(hdf5_name, 'r')

    # Get the model names, dropping the 'batch_code'
    # as it is not a model.
    models = sorted(hdf['/0'].keys())
    del(models[models.index('batch_code')])
    
    return models


def get_metadata(hdf5_name, model_name):
    """ Get the BOLD and DM metadata for <model> from the hdf file
    <name>. """

    # Open,
    hdf = h5py.File(hdf5_name, 'r')
    
    # and get its metadata.
    meta = {}
    meta['bold'] = hdf['/1/' + model_name + '/data/meta/bold'].value
    meta['dm'] = hdf['/1/' + model_name + '/data/meta/dm'].value

    return meta


def get_roi_names(hdf5_name):
    """ Returns a list of the roi names in <hdf5_name>. """
    
    return get_hdf_data(hdf5_name, '/batch_code')


def write_all_scores_as_df(hdf5_name, code):
    """ Using the data from <hdf5_name>, write all model fit scores 
    for all ROIs in a DataFrame. <code> should be the subject code. """
    from copy import deepcopy
    
    model_names = get_model_names(hdf5_name)
    roi_names = get_roi_names(hdf5_name)
    score_names = ['bic', 'aic', 'llf', 'r', 'r_adj', 'fvalue', 'f_pvalue']

    dataframe = []
        ## Score names will be the header
        ## of the dataframe

    for model in model_names:
        data = np.zeros((len(roi_names), len(score_names)))
            ## Will hold only the numerical scores
        for col, score in enumerate(score_names):
            path = '/'.join(['', model, score])
            data[:,col] = get_hdf_data(hdf5_name, path)
        
        for row, roi in zip(data, roi_names):
            # Build ech row of the dataframe...
            # Combine data with the metadata 
            # for that model/roi/etc
            meta = get_metadata(hdf5_name, model)        
            df_row = row.tolist() + [
                    code, str(roi), model, '_'.join(meta['dm'])]
            dataframe.append(df_row)

    # And write it out.
    header = score_names + ['sub', 'roi', 'model', 'dm']
    filename = hdf5_name.split('.')[0] +  '_scores.txt'
    fid = open(filename, 'w')
    writer = csv.writer(fid, delimiter='\t')
    writer.writerow(header)
    writer.writerows(dataframe)
    fid.close()

