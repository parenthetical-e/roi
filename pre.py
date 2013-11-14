""" A module for selecting voxels from within ROIs """
import numpy as np
import nibabel as nb
from roi.io import read_nifti


def combine4d(niftis):
    """Combine the list of nifti objects along their 4th axis.
    
    Note:
    ----
    * Assumes the nifti header for the first is nii is represenative of 
    the rest (use with care).
    * nibabel offers nibabel.concat_images() but it concats by
    creating a fifth axis, i.e., if nii1 was 10x10x10x10 and so
    was nii2 the concat_nii would be 10x10x10x10x2 instead
    of 10x10x10x20, the desired result.
    """

    # Init then join.
    combine_nifti = niftis.pop(0)
    for nifti in niftis:
        combine_nifti = join_time(combine_nifti, nifti)

    return combine_nifti


def join_time(nifti1, nifti2):
    """ Join two <nifti> objects along their 4th axis, time (i.e 
    volumes or TRs). Note: affline and header data is inherited from
    nifti1. """

    # Get the data for nifti1 and 2,
    data1 = nifti1.get_data()
    shape1 = data1.shape

    data2 = nifti2.get_data()
    shape2 = data2.shape

    joined = np.array([])     ## Init

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
    asnifti = nb.Nifti1Image(
            joined, affine=nifti1.get_affine(), header=nifti1.get_header())
    asnifti.update_header()

    return asnifti


def num_active_voxels(nifti):
    """ Returns the number of voxels in the first volume of <nifti>. """

    # Assume 3d, but if 4d 
    # only keep first vol. 
    vol = nifti.get_data()
    if vol.ndim > 3:
        vol = vol[...,0]

    return np.sum(vol > 0.01)


def mask(nifti, roi, standard=True):
    """ Mask and return the data in <nifti> with that in <roi> 
    (both should be nibabel obejcts). <roi> should be binary. 
    
    If <standard> is True (default) the q_form affine matrix is used.  
    If false, the s_form is used.
    
    For explanations of the q_form and s_form affine matrices in the 
    nifti1 format see Q19 at

        http://nifti.nimh.nih.gov/nifti-1/documentation/faq#Q2
    
    as well as
       
       http://nifti.nimh.nih.gov/dfwg/presentations/nifti-1-rationale 
    """

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

    nifti_data = nifti.get_data().astype('int16')
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
    roi_mask = roi.get_data().astype('int8') == 1
    print("{0} voxels in the mask.".format(np.sum(roi_mask)))

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
    neighborhood = np.abs(np.diag(nifti_affine)[0:3]) - np.abs(np.diag(roi_affine)[0:3])
    neighborhood[neighborhood < 1] = 1.0
        ## Any fractions or negative values should be set to zero
        ## as they are within the neighborhood

    # As _search_neighborhood() returns subarays
    # Needed an efficient way to concat them
    # row-wise, thus list.extend()
    matches = [] 
    [matches.extend(_search_neighborhood(
            nifti_std_index, xyz, neighborhood)) for xyz in roi_std_index]
    matches = np.array(matches)
        ## But in the end we need an array

    inv_nifti_affine = np.linalg.inv(nifti_affine)
    nifti_index_reduced = np.array([_affine_xyz(xyz, inv_nifti_affine) for 
            xyz in matches])
        ## Just invert the affine to get back
        ## to native space

    # Use n_i_reduced to make a mask...
    # Starting with zeros at every match 
    # put in a 1, finally convert to bool.
    vol_mask = np.zeros(nifti_shape, dtype=np.bool)
    for x, y, z in nifti_index_reduced:
        vol_mask[x, y, z] += True

    # -- 
    # Reduce the data from nifti 
    nifti_data_reduced = np.zeros(nifti_shape + (n_vol, ), dtype=np.int16)
    for vol in range(n_vol):
        nifti_data_reduced[vol_mask,vol] += nifti_data[vol_mask,vol] 

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
    
    diff_index_location = np.abs(index - location)
    n_mask = np.sum(diff_index_location <= neighborhood, axis=1) == 3
    
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
    # it is an index afterall.
    xyz_trans = np.int16(np.round(affine.dot(homo_xyz.transpose())))
        ## Rounding here is 1d nearest
        ## neighbor interpolation
        ## over each of the orthogonal
        ## axes (i.e. x, y, z).
    
    return xyz_trans[0:3]
        ## Dropping homogenous coords
    

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

