"""
Defines utility functions for saving png and hdf5 images.
"""
import imageio
import nibabel as nib
import numpy as np


def save_image_as_nii(image, filename):
    """
    Saves image as nii.gz file
    """
    assert filename.endswith('.nii.gz'), "incorrect extension"
    affine = np.diag([-1, -1, 1, 1])
    nii_img = nib.Nifti1Image(image.transpose(), affine)
    nib.save(nii_img, filename)


def save_image_as_png(image, filename):
    """
    Saves image as png files while preserving bit depth of the image
    """
    imageio.imwrite(filename, image)