"""
Defines utility functions for reading png and hdf5 images.
"""
import numpy as np
import imageio
import nibabel as nib


def read_image_nii(filename):
    return nib.load(filename).get_fdata().T

def read_image_png(file_name):
    image = np.array(imageio.imread(file_name))
    return image