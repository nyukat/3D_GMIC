"""
Module that define the core logic of GMIC
"""

import torch
import torch.nn as nn
import numpy as np
from src.utilities import tools
import src.modeling.modules as m


class GMIC3D(nn.Module):
    def __init__(self, parameters):
        super(GMIC3D, self).__init__()

        # save parameters
        self.experiment_parameters = parameters

        # construct networks
        # global network
        self.global_network = m.GlobalNetwork(self.experiment_parameters, self)
        self.global_network.add_layers()

        # aggregation function
        self.aggregation_function = m.TopTPercentAggregationFunctionFlattened(self.experiment_parameters, self)

        # detection module
        self.retrieve_roi_crops = m.RetrieveROIModule3D(self.experiment_parameters, self)

        # detection network
        self.local_network = m.LocalNetwork(self.experiment_parameters, self)
        self.local_network.add_layers()

        # MIL module
        self.attention_module = m.AttentionModule(self.experiment_parameters, self)
        self.attention_module.add_layers()

    def _convert_crop_position(self, crops_x_small, cam_size, x_original):
        """
        Function that converts the crop locations from cam_size to x_original
        :param crops_x_small: N, k*c, 2 numpy matrix
        :param cam_size: (h,w)
        :param x_original: N, C, H, W pytorch variable
        :return: N, k*c, 2 numpy matrix
        """
        # retrieve the dimension of both the original image and the small version
        h, w = cam_size
        _, _, H, W = x_original.size()

        # interpolate the 2d index in h_small to index in x_original
        top_k_prop_x = crops_x_small[:, :, 0] / h
        top_k_prop_y = crops_x_small[:, :, 1] / w
        # sanity check
        assert np.max(top_k_prop_x) <= 1.0, "top_k_prop_x >= 1.0"
        assert np.min(top_k_prop_x) >= 0.0, "top_k_prop_x <= 0.0"
        assert np.max(top_k_prop_y) <= 1.0, "top_k_prop_y >= 1.0"
        assert np.min(top_k_prop_y) >= 0.0, "top_k_prop_y <= 0.0"
        # interpolate the crop position from cam_size to x_original
        top_k_interpolate_x = np.expand_dims(np.around(top_k_prop_x * H), -1)
        top_k_interpolate_y = np.expand_dims(np.around(top_k_prop_y * W), -1)
        top_k_interpolate_2d = np.concatenate([top_k_interpolate_x, top_k_interpolate_y], axis=-1)
        return top_k_interpolate_2d

    def _retrieve_crop_3d(self, x_original_pytorch, crop_positions, crop_method, max_slice_numbers):
        """
        Function that takes in the original image and cropping position and returns the crops
        
        crop_positions contains all potential crop locations for all slices at each step.
        However, only the maximum crop among all slices is used at each step, indicated by max_slice_numbers.
        Therefore, for each step j, select only the true globally-maximum crop and ignore the rest.
        
        Assumes batch size of 1
        
        :param x_original_pytorch: PyTorch Tensor array (N,C,H,W)
        :param crop_positions:
        :return:
        """
        batch_size = 1
        num_slices, num_crops, _ = crop_positions.shape
        crop_h, crop_w = self.experiment_parameters["crop_shape"]

        output = torch.ones(
            (batch_size, num_crops, 1, crop_h, crop_w))  
        if self.experiment_parameters["half"]:
            output = output.half()
        if self.experiment_parameters["device_type"] == "gpu":
            output = output.cuda()
        for i in range(batch_size):
            for j in range(num_crops):
                tools.crop_pytorch_3d(x_original_pytorch[max_slice_numbers[j].item(), :, :, :],
                                   self.experiment_parameters["crop_shape"],
                                   crop_positions[max_slice_numbers[j].item(), j, :],
                                   output[i, j, :, :, :],
                                   method=crop_method)
        return output


    def forward(self, x_original):
        """
        :param x_original: N,C,D,H,W torch tensor 
        """
        N, C, num_slices, image_height, image_width = x_original.shape
        
        assert N == 1, "3D-GMIC is designed to work with batch size of 1 per GPU"
        assert C == 1, "Input is expected to be 1-channel image"
        
        # reshape the tensor so that the slice dimension is now batch dimension
        x_original = x_original.reshape(num_slices, 1, image_height, image_width)
        
        # global network
        h_g, self.saliency_map = self.global_network.forward(x_original)
        
        num_slices, num_classes, H, W = self.saliency_map.shape
        cam_size = (H, W)

        # calculate y_global
        saliency_map_flattened = self.saliency_map.permute(1,0,2,3).reshape(1, num_classes, -1)
        self.y_global = self.aggregation_function.forward(saliency_map_flattened, num_slices=num_slices) 

        # region proposal network
        self.intended_max_slice_numbers, self.max_slice_numbers, small_x_locations = self.retrieve_roi_crops.forward(x_original, cam_size, self.saliency_map)

        # convert crop locations that is on cam_size to x_original
        self.patch_locations = self._convert_crop_position(small_x_locations, cam_size, x_original) 

        # patch retriever
        crops_variable = self._retrieve_crop_3d(x_original, self.patch_locations, self.retrieve_roi_crops.crop_method, self.max_slice_numbers)

        # detection network
        batch_size, num_crops, num_slices_per_patch, I, J = crops_variable.size()
        assert batch_size == 1
        self.patches = crops_variable
        crops_variable_reshaped = crops_variable.view(batch_size * num_crops, num_slices_per_patch, I, J)
        h_crops = self.local_network.forward(crops_variable_reshaped).view(batch_size, num_crops, -1)

        # MIL module
        z, self.patch_attns, self.y_local = self.attention_module.forward(h_crops)

        # final output without using fusion branch
        self.final_prediction = 0.5 * self.y_global + 0.5 * self.y_local

        return self.final_prediction