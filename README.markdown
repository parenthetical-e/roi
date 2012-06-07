This is a set of functions for doing ROI analyses of fMRI data.  Using it
requires a decent working knowledge of python and numpy.  If you have that, it should be very striaght forward to use.  I hope.

It depends on numpy, scikits.statsmodels, nibabel, h5py, pil

It is further assumed that much of the preprocessing (coregistration, normalization, smoothing, etc) has been done in another package like SPM, AFNI, FSL, and so on.  

Also, fMRI data must be in the 4d nifit-1 (.nii) format.

# INTRODUCTION

Most the work is done by the Roi() class at roi.template.Roi(), which takes only the TR. This template class is designed to be just that, a template.  You must subclass it and setup various aspects before it does any useful work.

Here is a simple example of how to do that...

# TODO 

- implement pre
- test io, pre
- clean up/prep template
- how will top-level work?

# IT WILL WORK LIKE...

Do stuff to prep trials and data ... then:

    roi = Roi(code, TR, nifti, onsets, durations, data)

Behind the scenes setup a bunch of models...and:

    results = roi.run(code)
    

