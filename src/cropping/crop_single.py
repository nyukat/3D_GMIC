import argparse

import src.cropping.crop_mammogram as crop_mammogram
import src.utilities.pickling as pickling


def crop_single_mammogram(mammogram_path, horizontal_flip, view,
                          cropped_mammogram_path, metadata_path,
                          num_iterations, buffer_size):
    """
    Crop a single mammogram image
    """
    metadata_dict = dict(
        short_file_path=None,
        horizontal_flip=horizontal_flip,
        full_view=view,
        side=view[0],
        view=view[2:],
    )
    cropped_image_info = crop_mammogram.crop_mammogram_one_image(
        scan=metadata_dict,
        input_file_path=mammogram_path,
        output_file_path=cropped_mammogram_path,
        num_iterations=num_iterations,
        buffer_size=buffer_size,
    )
    metadata_dict["window_location"] = cropped_image_info[0]
    metadata_dict["rightmost_points"] = cropped_image_info[1]
    metadata_dict["bottommost_points"] = cropped_image_info[2]
    metadata_dict["distance_from_starting_side"] = cropped_image_info[3]
    pickling.pickle_to_file(metadata_path, metadata_dict)


def main():
    parser = argparse.ArgumentParser(description='Remove background of image and save cropped files')
    parser.add_argument('--mammogram-path', required=True)
    parser.add_argument('--view', required=True)
    parser.add_argument('--horizontal-flip', default="NO", type=str)
    parser.add_argument('--cropped-mammogram-path', required=True)
    parser.add_argument('--metadata-path', required=True)
    parser.add_argument('--num-iterations', default=100, type=int)
    parser.add_argument('--buffer-size', default=50, type=int)
    args = parser.parse_args()

    crop_single_mammogram(
        mammogram_path=args.mammogram_path,
        view=args.view,
        horizontal_flip=args.horizontal_flip,
        cropped_mammogram_path=args.cropped_mammogram_path,
        metadata_path=args.metadata_path,
        num_iterations=args.num_iterations,
        buffer_size=args.buffer_size,
    )


if __name__ == "__main__":
    main()