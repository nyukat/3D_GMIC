"""
Defines utility functions for managing the dataset.
"""
from src.constants import VIEWS


def unpack_exam_into_images(exam_list, cropped=False):
    """
    Turn exam_list into image_list for parallel functions which process each image separately.
    """
    image_list = []
    for i, exam in enumerate(exam_list):
        for view in VIEWS.LIST:
            for j, image in enumerate(exam[view]):
                image_dict = dict(
                    short_file_path=image,
                    horizontal_flip=exam['horizontal_flip'],
                    full_view=view,
                    side=view[0],
                    view=view[2:],
                )
                if cropped:
                    image_dict["window_location"] = exam['window_location'][view][j]
                    image_dict["rightmost_points"] = exam['rightmost_points'][view][j]
                    image_dict["bottommost_points"] = exam['bottommost_points'][view][j]
                    image_dict["distance_from_starting_side"] = exam['distance_from_starting_side'][view][j]
                image_list.append(image_dict)
    return image_list


def add_metadata(exam_list, additional_metadata_name, additional_metadata_dict):
    """
    Includes new information about images into exam_list
    """
    for exam in exam_list:
        assert additional_metadata_name not in exam, "this metadata is already included"
        exam[additional_metadata_name] = dict()
        for view in VIEWS.LIST:
            exam[additional_metadata_name][view] = []
            for j, image in enumerate(exam[view]):
                exam[additional_metadata_name][view].append(additional_metadata_dict[image])