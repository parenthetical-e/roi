""" A module for selecting voxels from within ROIs """
import numpy as np
import nibabel as nb


def join_time(nifti1, nifti2):
    """ Join two <nifti> objects along their 4th axis, time (i.e 
    volumes or TRs). Note: affline and header data is lost. """

    # Get the data for nifti1 and 2,
    data1 = nifti1.get_data()
    shape1 = data1.shape

    data2 = nifti2.get_data()
    shape2 = data2.shape

    joined = np.array()     ## Init

    # and join it.
    try:
        joined = np.append(data1, data2, 3)
    except ValueError:
        # If nifti1 or 2 is only 3d the above errors
        # so add a dummy 4th,
        if len(shape1) == 3:
            data1.reshape((data1.shape + (1,)))

        if len(shape2) == 3:
            data2.reshape((data2.shape + (1,)))
        
        # then try again.
        joined = np.append(data1, data2, 3)

    # Convert to a nifti object
    return nb.Nifti1Image(joined)


def mask(nifti, roi, standard=True):
    """ Mask and returm the data in <nifti> with that in <roi> 
    (both should be nibabel obejcts). <roi> should be binary. 
    
    If <standard> is True (default) the q_form affine matrix is used.  
    If false, the s_form is used.
    
    For explanations of the q_form and s_form affine matrices in the 
    nifti1 format see Q19 at

        http://nifti.nimh.nih.gov/nifti-1/documentation/faq#Q2
    
    as well as
       
       http://nifti.nimh.nih.gov/dfwg/presentations/nifti-1-rationale """

    # -- 
    # Create nifti and roi meta-data
    nifti_head = nifti.get_header()
    roi_head = roi.get_header()

    if standard:
        roi_affine = roi_head.get_qform()
        nifti_affine = nifti_head.get_qform()
    else:
        roi_affine = roi_head.get_sform()        
        nifti_affine = nifti_head.get_sform()

    nifti_data = nifti.get_data()
    nifti_shape = nifti.shape

    # --
    # Deal with nifti_data being 3d
    # *or* 4d
    if len(nifti_shape) > 3:
        # When 4d:
        n_vol = nifti_shape[3]
        nifti_shape = nifti_shape[0:3]
            ## We only want x,y,z -> 0,1,2
    else:
        # When 3d:
        nifti_data = nifti_data[...,np.newaxis]
            ## We need to add a empty 4th d
        n_vol = 1
            ## so this makes sense.

    # --
    # Find only voxels that are 1
    # in the roi native space
    # then convert these to standard space
    roi_mask = roi.get_data() == 1
    roi_index = _native_index(roi_mask)
    roi_index_filtered = np.array(
            [
                np.array([x, y, z]) for x, y, z in roi_index if 
                    roi_mask[x, y, z]
            ])

    roi_std_index = np.array([_affine_xyz(xyz, roi_affine) for 
            xyz in roi_index_filtered])

    # Convert nifti to standard space too
    nifti_std_index = _affine_index(nifti_data, nifti_affine)
   
    # --
    # Find neighborhoods where nifti and roi overlap
    # (in standard space).
    neighborhood = np.abs(np.diag(nifti_affine)[0:3])
    neighborhood[neighborhood < 1] = 0
        ## Any fractions should be set to zero
        ## as they are within the neighborhood
        ## by default

    print("size of: roi_index: {0}, roi_index_filtered: {1}, roi_std_index: {2}, nifti_std_index: {3}, neighborhood: {4}".format(len(roi_index), len(roi_index_filtered), len(roi_std_index), nifti_std_index.shape, neighborhood))

    # As _search_neighborhood() returns subarays
    # Needed an efficient way to concat them
    # row-wise, thus list.extend()
    matches = list()
    [matches.extend(_search_neighborhood(
            nifti_std_index, xyz, neighborhood)) for xyz in roi_std_index]
    matches = np.array(matches)
        ## But in the end we need an array

    print("Shape of matches: {0}".format(matches.shape))

    inv_nifti_affine = np.linalg.inv(nifti_affine)
    nifti_index_reduced = np.array([_affine_xyz(xyz, inv_nifti_affine) for 
            xyz in matches])
        ## Just invert the affine to get back
        ## to native space
    
    print("inv_roi_affine: {0}".format(inv_nifti_affine))
    print("size: nifti_index_reduced: {0}".format(len(nifti_index_reduced)))

    # Use n_i_reduced to make a mask...
    # Starting with zeros at every match 
    # put in a 1, finally convert to bool.
    vol_mask = np.zeros(nifti_shape)
    for x, y, z in nifti_index_reduced:
        vol_mask[x, y, z] = 1

    print vol_mask
    vol_mask = vol_mask == 1
    print("Shape: vol_mask: {0}".format({vol_mask.shape}))

    # -- 
    # Reduce the data from nifti 
    nifti_data_reduced = np.zeros(nifti_shape + (n_vol, ))
    for vol in range(n_vol):
        nifti_data_reduced[vol_mask,vol] = nifti_data[vol_mask,vol]  
 
    # -- 
    # Return a nifti object with
    # proper meta-data
    return nb.Nifti1Image(
            nifti_data_reduced, nifti_affine, nifti.get_header())


def _search_neighborhood(index, location, neighborhood):
    """ Search index (a 2d column oriented array), where each row
    is a set of x,y,z coordinates for <location> (an x,y,z sequence) that
    is within the <neighborhood> of location. 
    
    Returns matching indices. """
    
    index = np.asarray(index)
    location = np.asarray(location)
    neighborhood = np.abs(np.asarray(neighborhood))
    
    n_mask = np.sum(np.abs(index - location) <= neighborhood, axis=1) == 3

    return index[n_mask,:]
    
        

def _native_index(data):
    """ Create an index for the given <shape>. Each set of x, y and z 
    coordinates are stored in a tuple, returns a list of these tuples.
    """

    data = np.asarray(data)

    shape = data.shape
    if len(shape) > 3:
        shape = shape[0:3]
    
    # np.indices maintains the shape of the orginal
    # which would be a pain to iterate over
    # so it is flattened.
    x_initial, y_initial, z_initial = np.indices(shape)
    x_flat = x_initial.flatten()
    y_flat = y_initial.flatten()
    z_flat = z_initial.flatten()

    # Make tuple-sets of the indices,
    # again simplfying iteration
    return np.array([x_flat, y_flat, z_flat]).transpose()
        ## Want col oriented for easy iteration
    

def _affine_index(data, affine):
    """ Returns a real-world (i.e. millimeter scale) index
    for the fMRI data stored in <data>, a 3 or 4d array. """
    
    data = np.asarray(data)     ## Just in case
    i_native = _native_index(data)
    
    return np.array([_affine_xyz(xyz, affine) for xyz in i_native])


def _affine_xyz(xyz, affine):
    """ Apply a <affine> transform (array-like) to
    x_coord, y_coord and z_coord, index values for some some unknown
    3d space (probably an fMRI data volume). 
    
    Returning the transformed values. 
    
    Note: this function is intended to tranforming indexes only,
    as a result it uses nearest neighbor interpolation.  For more 
    flexible, general purpose, Affine transforms see the Scipy.ndimage
    as well as PIL (Python Imaging Library). """
 
        # Make sure the affine looks right...
    if affine.shape != (4, 4):
        raise ValueError("affine matrix must be square, of rank 4.")
    if np.sum(affine[3,:]) != 1:
        raise ValueError("affine matrix is not in homogenous coordinates")

    x_coord, y_coord, z_coord = xyz
    homo_xyz = np.array([x_coord, y_coord, z_coord, 1])
        ## Adding the 1 so xyz are in homogenous coordinates:
        ## 
        ## Mortenson, Michael E. (1999). Mathematics for Computer Graphics
        ## Applications. Industrial Press Inc. p. 318. 
        ## 
        ## See also the Wiki entry
        ## http://en.wikipedia.org/wiki/Transformation_matrix
        ## the affine section.

    # The transform, at last
    # and convert to int, 
    # it is an index point afterall.
    xyz_trans = np.int16(np.round(affine.dot(homo_xyz.transpose())))
        ## Rounding here is 1d nearest
        ## neighbor interpolation
        ## over each of the orthogonal
        ## axes (i.e. x, y, z).
    
    return xyz_trans[0:3]
        ## Dropping homogenous rows
    

# Use index for each to mask.
#def top_t(nifti, tmap, percent):
#    pass

#def top_var(nifti, percent):
#    pass

#def top_info(nifti, percent):
#    pass

#def pca(nifti, num):
#    pass

#def ica(nifti, num):
#    pass
