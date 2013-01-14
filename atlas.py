""" Use atlases to create ROI masks. """
import os
import nibabel as nb
import numpy as np
import roi
from zipfile import ZipFile
from roi.io import write_nifti, read_nifti
    

def get_roi(atlas, name):
    """ Get the an roi (a nifti1 object) by <name> from <atlas>. """
    
    # Set the path for the named roi
    # and try to open it.
    path = os.path.join(roi.__path__[0], 'atlases', atlas, 'rois', name)
    try:
        nifti = read_nifti(path)

        # If name can't be loaded you may need to run
        # create_rois, tell the user that.
    except IOError, err:
        print("Could not load <name>. Try create_rois() or open_atlases()")
        raise IOError(err)
    
    return nifti


def open_atlases():
    """ Uncompress the atlases file. """
    
    if not os.path.exists(os.path.join(roi.__path__[0], 'atlases')):
        print("Unzipping atlases in {0}.".format(
                os.path.join(roi.__path__[0], 'atlases')))
        
        unzme = ZipFile(os.path.join(roi.__path__[0], 'atlases.zip'))
        unzme.extractall(path=roi.__path__[0])


def create_rois(atlas, base, legend):
    """ Creates ROI masks for from given <atlas> using <base> (a 4d nifit1 
    file containg all rois.  A <legend>.txt file mapping roi codes to
    names is also needed, formatted as 1:Hippocampus. """

    # setup pathing, read in the process the base
    path = os.path.join(roi.__path__[0], 'atlases')
    loni = read_nifti(os.path.join(path, atlas, base))

    # Need these for writing the rois out later
    header = loni.get_header()
    affline = loni.get_affine()
    data = loni.get_data()

    # And get then the txt file that is its legend.
    fhandle = open(os.path.join(path, legend))
    fdata = [line.strip().split(':') for line in fhandle]

    # Store the legend data in a dict keyed
    # on the ROI names.
    legend_dict = {}
    for col1, col2 in fdata:
        legend_dict.update({col2 : int(col1)})
    
    # Loop over the legend, creating and 
    # saving a nii file for each item.
    roi_path = os.path.join(path, atlas, 'rois')
    if not os.path.exists(roi_path):
        os.mkdir(roi_path)

    for key, val in legend_dict.items():
        # Mask based on v, the index for he current
        # roi then create the binary roi.
        mask = np.zeros(data.shape, dtype=np.uint8)
            ## nifti1 needs unsigned integers

        mask[data == val] = 1

        # Finally, create a nifti object and write 
        # it using the key from legend as a name.
        niftiname = key + '.nii'
        nifti = nb.Nifti1Image(mask, affline, header)
        nifti.update_header()  
            ## Not sure this does anything,
            ## but just in case

        write_nifti(nifti, os.path.join(path, atlas, 'rois', niftiname))