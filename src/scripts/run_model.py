"""
Script that executes the model pipeline.
"""

import argparse
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import tqdm
import cv2
import matplotlib.cm as cm
import imageio
import copy
from collections import defaultdict

from src.utilities import pickling, tools
from src.modeling import gmic3d as gmic3d
from src.data_loading import loading
from src.constants import VIEWS, PERCENT_T_DICT, TOP_K_DICT

def visualize_example(input_img, saliency_maps,
                      sampled_slice_numbers, patch_locations, patch_img, patch_attentions,
                      save_dir, parameters):
    """
    Function that visualizes the saliency maps for an example
    
    sampled_slice_numbers shape: (K, 1)
    patch_locations       shape: (num_slices, K, 2) 
    patch_img             shape: (1, K, 1, 256, 256) 
    patch_attentions      shape: (K,)
    """
    # colormap lists
    _, _, h, w = saliency_maps.shape
    _, _, num_slices, H, W = input_img.shape
    
    input_img_min = input_img.min()
    input_img_max = input_img.max()

    # set up colormaps for benign and malignant
    alphas = np.abs(np.linspace(0, 0.95, 259))
    alpha_green = copy.copy(plt.cm.get_cmap('Greens'))
    alpha_green._init()
    alpha_green._lut[:, -1] = alphas
    alpha_red = copy.copy(plt.cm.get_cmap('Reds'))
    alpha_red._init()
    alpha_red._lut[:, -1] = alphas
    
    # set up colormaps for patch map
    patch_cmap = copy.copy(cm.YlGnBu)
    patch_cmap.set_under('w', alpha=0)
    
    # get the patches that are actually used in 3D-GMIC
    patches_dict = defaultdict(list)
    for patch_idx in range(parameters["K"]):
        patches_dict[sampled_slice_numbers[patch_idx][0]].append(patch_idx)

    for slice_idx in range(num_slices):
        # create visualization template
        total_num_subplots = 4
        figure = plt.figure(figsize=(8, 3))
        # input image + segmentation map
        subfigure = figure.add_subplot(1, total_num_subplots, 1)
        subfigure.imshow(input_img[0, 0, slice_idx, :, :], aspect='equal', cmap='gray', clim=[input_img_min, input_img_max])
        subfigure.set_title("input image")
        subfigure.axis('off')

        # patch map
        subfigure = figure.add_subplot(1, total_num_subplots, 2)
        subfigure.imshow(input_img[0, 0, slice_idx, :, :], aspect='equal', cmap='gray', clim=[input_img_min, input_img_max])
        if slice_idx in patches_dict:
            crop_mask = tools.get_crop_mask(
                patch_locations[slice_idx, patches_dict[slice_idx], :],
                parameters["crop_shape"], (H, W),
                "upper_left")
            subfigure.imshow(crop_mask, alpha=0.7, cmap=patch_cmap, clim=[0.9, 1])
        subfigure.set_title("patch map")
        subfigure.axis('off')

        # class activation maps
        subfigure = figure.add_subplot(1, total_num_subplots, 4)
        subfigure.imshow(input_img[0, 0, slice_idx, :, :], aspect='equal', cmap='gray', clim=[input_img_min, input_img_max])
        resized_cam_malignant = cv2.resize(saliency_maps[slice_idx,1,:,:], (W, H), interpolation=cv2.INTER_NEAREST)
        subfigure.imshow(resized_cam_malignant, cmap=alpha_red, clim=[0.0, 1.0])
        subfigure.set_title("SM: malignant")
        subfigure.axis('off')

        subfigure = figure.add_subplot(1, total_num_subplots, 3)
        subfigure.imshow(input_img[0, 0, slice_idx, :, :], aspect='equal', cmap='gray', clim=[input_img_min, input_img_max])
        resized_cam_benign = cv2.resize(saliency_maps[slice_idx,0,:,:], (W, H), interpolation=cv2.INTER_NEAREST)
        subfigure.imshow(resized_cam_benign, cmap=alpha_green, clim=[0.0, 1.0])
        subfigure.set_title("SM: benign")
        subfigure.axis('off')

        plt.savefig(os.path.join(save_dir, f'{slice_idx}.png'), bbox_inches='tight', format="png", dpi=300)
        plt.close()
        
    frames_list = []
    for slice_idx in range(num_slices):
        frames_list.append(imageio.imread(os.path.join(save_dir, f'{slice_idx}.png')))
    imageio.mimsave(os.path.join(save_dir, f'all_slices_vis.gif'), frames_list, duration=0.1)

    # crops
    total_num_subplots = parameters["K"]
    figure = plt.figure(figsize=(total_num_subplots+2, 2))
    for crop_idx in range(parameters["K"]):
        subfigure = figure.add_subplot(1, total_num_subplots, 1 + crop_idx)
        subfigure.imshow(patch_img[0, crop_idx, 0, :, :], cmap='gray', alpha=.8, interpolation='nearest',
                         aspect='equal')
        subfigure.axis('off')
        # crops_attn can be None when we only need the left branch + visualization
        subfigure.set_title("$\\alpha_{{{0}}} = ${1:.2f}".format(crop_idx, patch_attentions[crop_idx]))
        
    plt.savefig(os.path.join(save_dir, 'patches.png'), bbox_inches='tight', format="png", dpi=200)
    plt.close()


def fetch_cancer_label_by_view(view, cancer_label):
    """
    Function that fetches cancer label using a view
    """
    if view in ["L-CC", "L-MLO"]:
        return cancer_label["left_benign"], cancer_label["left_malignant"]
    elif view in ["R-CC", "R-MLO"]:
        return cancer_label["right_benign"], cancer_label["right_malignant"]


def run_model(model, exam_list, parameters, turn_on_visualization):
    """
    Run the model over images in sample_data.
    Save the predictions as csv and visualizations as png.
    """
    if (parameters["device_type"] == "gpu") and torch.has_cudnn:
        device = torch.device("cuda:{}".format(parameters["gpu_number"]))
    else:
        device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    # initialize data holders
    pred_dict = {"image_index": [], "benign_pred": [], "malignant_pred": [],
     "benign_label": [], "malignant_label": []}
    with torch.no_grad():
        # iterate through each exam
        for datum in tqdm.tqdm(exam_list):
            for view in VIEWS.LIST:
                short_file_path = datum[view][0]
                # load image
                loaded_image = loading.load_dbt_image(
                    image_path=os.path.join(parameters["image_path"], short_file_path + ".nii.gz"),
                    view=view,
                    horizontal_flip=datum["horizontal_flip"],
                )
                loaded_image = loading.process_dbt_image(loaded_image, view, datum["best_center"][view][0])
                num_slices, image_height, image_width = loaded_image.shape
                # convert python 3D array into 5D torch tensor in (N,C,D,H,W) format
                loaded_image = loaded_image.reshape(
                    1, 1, num_slices, image_height, image_width
                )
                tensor_batch = torch.Tensor(loaded_image).to(device)
                # forward propagation
                output = model(tensor_batch)
                pred_numpy = output.data.cpu().numpy()
                benign_pred, malignant_pred = pred_numpy[0, 0], pred_numpy[0, 1]
                
                # save visualization
                if turn_on_visualization:
                    saliency_maps = model.saliency_map.data.cpu().numpy()
                    sampled_slice_numbers = model.max_slice_numbers
                    patch_locations = model.patch_locations
                    patch_imgs = model.patches.data.cpu().numpy()
                    patch_attentions = model.patch_attns[0, :].data.cpu().numpy()
                    save_dir = os.path.join(parameters["output_path"], "visualization", short_file_path)
                    os.makedirs(save_dir, exist_ok=True)
                    visualize_example(
                        loaded_image, saliency_maps,
                        sampled_slice_numbers, patch_locations, patch_imgs, patch_attentions,
                        save_dir, parameters
                    )
                # propagate holders
                benign_label, malignant_label = fetch_cancer_label_by_view(view, datum["cancer_label"])
                pred_dict["image_index"].append(short_file_path)
                pred_dict["benign_pred"].append(benign_pred)
                pred_dict["malignant_pred"].append(malignant_pred)
                pred_dict["benign_label"].append(benign_label)
                pred_dict["malignant_label"].append(malignant_label)
    return pd.DataFrame(pred_dict)


def run_single_model(model_path, data_path, parameters, turn_on_visualization):
    """
    Load a single model and run on sample data
    """
    # construct model
    model = gmic3d.GMIC3D(parameters)
    # load parameters
    if parameters["device_type"] == "gpu":
        model.load_state_dict(torch.load(model_path), strict=True)
    else:
        model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
    # load metadata
    exam_list = pickling.unpickle_from_file(data_path)
    # run the model on the dataset
    output_df = run_model(model, exam_list, parameters, turn_on_visualization)
    return output_df


def start_experiment(model_path, data_path, output_path, model_index, parameters, turn_on_visualization):
    """
    Run the model on sample data and save the predictions as a csv file
    """
    # make sure model_index is valid
    valid_model_index = ["1", "2", "3", "4", "5", "ensemble"]
    assert model_index in valid_model_index, "Invalid model_index {0}. Valid options: {1}".format(model_index, valid_model_index)
    # create directories
    os.makedirs(output_path, exist_ok=True)
    if turn_on_visualization:
        os.makedirs(os.path.join(output_path, "visualization"), exist_ok=True)
    # do the average ensemble over predictions
    if model_index == "ensemble":
        output_df_list = []
        for i in range(1,6):
            single_model_path = os.path.join(model_path, "sample_model_{0}.p".format(i))
            # set percent_t for the model
            parameters["percent_t"] = PERCENT_T_DICT[str(i)]
            parameters["K"] = TOP_K_DICT[str(i)]
            # only do visualization for the first model
            need_visualization = i==1 and turn_on_visualization
            current_model_output = run_single_model(single_model_path, data_path, parameters, need_visualization)
            output_df_list.append(current_model_output)
        all_prediction_df = pd.concat(output_df_list)
        output_df = all_prediction_df.groupby("image_index").apply(lambda rows: pd.Series({"benign_pred":np.nanmean(rows["benign_pred"]),
                      "malignant_pred": np.nanmean(rows["malignant_pred"]),
                      "benign_label": rows.iloc[0]["benign_label"],
                      "malignant_label": rows.iloc[0]["malignant_label"],
                      })).reset_index()
    else:
        # set percent_t for the model
        parameters["percent_t"] = PERCENT_T_DICT[model_index]
        parameters["K"] = TOP_K_DICT[model_index]
        single_model_path = os.path.join(model_path, "sample_model_{0}.p".format(model_index))
        output_df = run_single_model(single_model_path, data_path, parameters, turn_on_visualization)

    # save the predictions
    output_df.to_csv(os.path.join(output_path, "predictions.csv"), index=False, float_format='%.4f')



def main():
    # retrieve command line arguments
    parser = argparse.ArgumentParser(description='Run 3D-GMIC on the sample data')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--image-path', required=True)
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--device-type', default="cpu", choices=['gpu', 'cpu'])
    parser.add_argument("--gpu-number", type=int, default=0)
    parser.add_argument("--model-index", type=str, default="1")
    parser.add_argument("--visualization-flag", action="store_true", default=False)
    parser.add_argument("--half", action="store_true", default=False)
    args = parser.parse_args()

    parameters = {
        "device_type": args.device_type,
        "gpu_number": args.gpu_number,
        "image_path": args.image_path,
        "segmentation_path": None,
        "output_path": args.output_path,
        # model related hyper-parameters
        "crop_shape": (256, 256),
        "post_processing_dim":256,
        "num_classes":2,
        "use_v1_global":False, 
        "half": False,
        "norm_class": 'group', # GroupNorm in GlobalNetwork
        "num_groups": 8, # GroupNorm in GlobalNetwork
        "saliency_nonlinearity": 'tanh_relu',
    }
    start_experiment(
        model_path=args.model_path,
        data_path=args.data_path,
        output_path=args.output_path,
        model_index=args.model_index,
        parameters=parameters,
        turn_on_visualization=args.visualization_flag,
    )

if __name__ == "__main__":
    main()
