# Copyright 2020 Toyota Research Institute.  All rights reserved.
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from itertools import groupby




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
# #128 256 3

            warped_image = _read_image(self.files['warped_image_paths'][idx])
            homography = np.array(self.files['homography'][idx])
            sample = {'image': image, 'warped_image': warped_image, 'homography': homography, 'index' : idx}


            transform = transforms.ToTensor()
            for key in ['image', 'warped_image']:
                sample[key] = transform(sample[key]).type('torch.FloatTensor')

            return sample
        except Exception as e:
            print(e)


def check_each_image_has_one_correspondance(grouped_images):
    for i in grouped_images:
        assert len(i) == 2, f'Error with the image(s): {i}'

class PatchesDataset_jj2(Dataset):
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
        pts1_set = []
        pts2_set = []
        file_ext = '.png'
        path4opt_imgs = "/home/jinjing/Projects/data_old/new_data/output/paired_opt/"#"/home/jinjing/Projects/data/final_ori_imgs/"



        images_paths = sorted(os.listdir(base_path))
        grouped_images = [list(i) for j, i in groupby(images_paths,
                        lambda img_name: img_name.split("_")[0])]




        check_each_image_has_one_correspondance(grouped_images)
        # number_of_images = len(grouped_images)
        # counter = 1
        for i, j in grouped_images:
            img1_path = str(Path(base_path, i))
            img2_path = str(Path(base_path, j))
            npz_path = path4opt_imgs+i[:-4]+"/GT_H_pts.npz"
            GT_H_pts = np.load(npz_path)
            GT_H = GT_H_pts["H"]
            pts1 = GT_H_pts["pts1"]
            pts2 = GT_H_pts["pts2"]

            homographies.append(GT_H)
            image_paths.append(img1_path)
            warped_image_paths.append(img2_path)
            pts1_set.append(pts1)
            pts2_set.append(pts2)













        self.files = {'image_paths': image_paths, 'warped_image_paths': warped_image_paths, 'homography': homographies,'pts1_set':pts1_set,'pts2_set':pts2_set}


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
            pts1 = np.array(self.files['pts1_set'][idx])
            pts2 = np.array(self.files['pts2_set'][idx])
            sample = {'image': image, 'warped_image': warped_image, 'homography': homography, 'index' : idx,'pts1':pts1,'pts2':pts2}


            transform = transforms.ToTensor()
            for key in ['image', 'warped_image']:

                sample[key] = transform(sample[key].astype(np.float32)).type('torch.FloatTensor')#torch.FloatTensor  320 3 240
                # cv2.imshow("",np.moveaxis(np.array(sample[key]).astype(np.uint8), 0, -1))
                # cv2.imshow("",np.moveaxis(np.array(sample[key]).astype(np.uint8), 0, -1))
                # cv2.waitKey(0)


            return sample
        except Exception as e:
            print(e)

