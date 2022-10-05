import numpy as np
from src.constants import VIEWS
import imageio
import src.data_loading.augmentations as augmentations
from src.utilities.reading_images import read_image_nii
from src.constants import INPUT_SIZE_DICT

def flip_image_last_dim(image, view, horizontal_flip):
    """
    If training mode, makes all images face right direction.
    In medical, keeps the original directions unless horizontal_flip is set.
    """
    if horizontal_flip == 'NO':
        if VIEWS.is_right(view):
            image = image[...,::-1]
    elif horizontal_flip == 'YES':
        if VIEWS.is_left(view):
            image = image[...,::-1]

    return image


def standard_normalize_single_image(image):
    """
    Standardizes an image in-place 
    """
    image -= np.mean(image)
    image /= np.maximum(np.std(image), 10**(-5))


def load_dbt_image(image_path, view, horizontal_flip):
    """
    Loads a png or hdf5 image as floats and flips according to its view.
    """
    if image_path.endswith(".nii.gz"):
        image = read_image_nii(image_path)
    else:
        raise RuntimeError()
    image = image.astype(np.float32)
    image = flip_image_last_dim(image, view, horizontal_flip)
    return image


def process_dbt_image(image, view, best_center):
    """
    Applies augmentation window with random noise in location and size
    and return normalized cropped image.
    """
    num_slices = image.shape[0]
    
    result_buffer = np.zeros([num_slices, *INPUT_SIZE_DICT[view]], dtype=np.float32)
    
    for slice_idx in range(num_slices):
        # When no randomness is applied, it is ok to crop each slice separately
        result_buffer[slice_idx], _ = augmentations.random_augmentation_best_center(
            image=image[slice_idx],
            input_size=INPUT_SIZE_DICT[view],
            random_number_generator=np.random.RandomState(0),
            best_center=best_center,
            view=view
        )

    # Normalize the entire 3D image at once
    standard_normalize_single_image(result_buffer)

    return result_buffer





