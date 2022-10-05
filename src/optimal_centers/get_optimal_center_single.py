"""
Runs search_windows_and_centers.py and extract_centers.py in the same directory
"""
import argparse

import src.utilities.pickling as pickling
import src.utilities.reading_images as reading_images
import src.optimal_centers.get_optimal_centers as get_optimal_centers


def get_optimal_center_dbt_single(cropped_mammogram_path, metadata_path):
    """
    Get optimal center for single example
    """
    metadata = pickling.unpickle_from_file(metadata_path)
    image = reading_images.read_image_nii(cropped_mammogram_path)
    optimal_center = get_optimal_centers.extract_center(metadata, image.max(0))
    metadata["best_center"] = optimal_center
    pickling.pickle_to_file(metadata_path, metadata)


def main():
    parser = argparse.ArgumentParser(description='Compute and Extract Optimal Centers')
    parser.add_argument('--cropped-mammogram-path', required=True)
    parser.add_argument('--metadata-path', required=True)
    parser.add_argument('--num-processes', default=20)
    args = parser.parse_args()
    get_optimal_center_dbt_single(
        cropped_mammogram_path=args.cropped_mammogram_path,
        metadata_path=args.metadata_path,
    )


if __name__ == "__main__":
    main()