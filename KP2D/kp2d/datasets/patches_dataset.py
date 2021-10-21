# Copyright 2020 Toyota Research Institute.  All rights reserved.
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset




class PatchesDataset_jj(Dataset):
    """
    HPatches dataset class.
    Note: output_shape = (output_width, output_height)
    Note: this returns Pytorch tensors, resized to output_shape (if specified)
    Note: the homography will be adjusted according to output_shape.

    Parameters
    ----------
    root_dir : str
        Path to the dataset
    use_color : bool
        Return color images or convert to grayscale.
    data_transform : Function
        Transformations applied to the sample
    output_shape: tuple
        If specified, the images and homographies will be resized to the desired shape.
    type: str
        Dataset subset to return from ['i', 'v', 'all']:
        i - illumination sequences
        v - viewpoint sequences
        all - all sequences
    """
    def __init__(self, root_dir, use_color=True, data_transform=None, output_shape=None, type='all'):

        super().__init__()
        self.type = type
        self.root_dir = root_dir
        self.data_transform = data_transform
        self.output_shape = output_shape
        self.use_color = use_color
        base_path = Path(root_dir)  #dirs for transformed imgs of each img_ori
        folder_paths = [x for x in base_path.iterdir() if x.is_dir()]
        image_paths = []
        warped_image_paths = []
        homographies = []
        file_ext = '.png'
        path4ori_imgs = "/home/jinjing/Projects/data/final_ori_imgs/"


        # a = np.loadtxt("/home/jinjing/Projects/data/new_data/output/i_haha_jj_oo/H_1_2")# 3*3 nd array  float64
        for trans_dir_all in folder_paths:
            ori_name = str(trans_dir_all).split("/")[-1]
            for trans_dir_sig in trans_dir_all.iterdir():
                num_images = 6  # for each kind of trans we have 6
                H_dir = np.load(str(trans_dir_sig)+"/GT_H_mat.npz")

                for trans_name in [f for f in os.listdir(trans_dir_sig) if f.endswith('.png')]:
                    # print("now its the ",str(trans_dir_sig)+"/"+trans_name+file_ext)
                    trans_name = trans_name[:-4]
                    image_paths.append(path4ori_imgs+ori_name+file_ext)
                    warped_image_paths.append(str(trans_dir_sig)+"/"+trans_name+file_ext)  # use Path() can avoid manually adding "/"
                    H_mat = H_dir[trans_name]
                    # print(trans_name)
                    homographies.append(H_mat)

            # for i in range(2, 2 + num_images):
            #     image_paths.append(str(Path(trans_dir_all, "1" + file_ext)))  # the ori img
            #     warped_image_paths.append(str(Path(trans_dir_all, str(i) + file_ext))) # the warped img
            #     # H_array_33 =
            #     homographies.append(np.loadtxt(str(Path(trans_dir_all, "H_1_" + str(i)))))  # the H mat








        self.files = {'image_paths': image_paths, 'warped_image_paths': warped_image_paths, 'homography': homographies}


    def __len__(self):
        return len(self.files['image_paths'])

    def __getitem__(self, idx):
        try:
            def _read_image(path):
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if self.use_color:
                    return img
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                return gray
            image = _read_image(self.files['image_paths'][idx])
            warped_image = _read_image(self.files['warped_image_paths'][idx])
            homography = np.array(self.files['homography'][idx])
            sample = {'image': image, 'warped_image': warped_image, 'homography': homography, 'index' : idx}


            transform = transforms.ToTensor()
            for key in ['image', 'warped_image']:
                sample[key] = transform(sample[key]).type('torch.FloatTensor')

            return sample
        except Exception as e:
            print(e)

