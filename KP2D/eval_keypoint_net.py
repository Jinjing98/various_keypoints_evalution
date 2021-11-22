# Copyright 2020 Toyota Research Institute.  All rights reserved.
# Example usage: python scripts/eval_keypoint_net.sh --pretrained_model /data/models/kp2d/v4.pth --input_dir /data/datasets/kp2d/HPatches/

import argparse
import os
import pickle
import random
import subprocess

import cv2
import numpy as np
import torch
from PIL import Image
from termcolor import colored
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import sys


sys.path.append('..')  #  this is important for same level import
# #https://blog.csdn.net/songbinxu/article/details/80289489
# #https://cloud.tencent.com/developer/article/1803921
# #https://www.jianshu.com/p/5a02285bb111

from kp2d.datasets.patches_dataset import PatchesDataset_jj,PatchesDataset_jj2 #PatchesDataset
from  kp2d.evaluation.evaluate import evaluate_keypoint_net_jj,evaluate_keypoint_net_jj2#evaluate_keypoint_net
from  kp2d.networks.keypoint_net import KeypointNet


def main():
    parser = argparse.ArgumentParser(
        description='Script for KeyPointNet testing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pretrained_model", type=str, help="pretrained model path",
                        default="/home/jinjing/Projects/keypoints_comparision/git_src_code/kp2d/pretrained_models/v4.ckpt")
    parser.add_argument("--input_dir", type=str, help="Folder containing input images",
                        default="/home/jinjing/Projects/data/out_imgs/")

    args = parser.parse_args()
    checkpoint = torch.load(args.pretrained_model)
    model_args = checkpoint['config']['model']['params']

    # Create and load disp net
    keypoint_net = KeypointNet(use_color=model_args['use_color'],
                               do_upsample=model_args['do_upsample'],
                               do_cross=model_args['do_cross'])
    keypoint_net.load_state_dict(checkpoint['state_dict'])
    keypoint_net = keypoint_net.cuda()
    keypoint_net.eval()
    print('Loaded KeypointNet from {}'.format(args.pretrained_model))
    print('KeypointNet params {}'.format(model_args))

    eval_params = [
                {  'top_k': 100, "res":(128,256), 'conf_threshold':0},
                {  'top_k': 200, "res":(128,256), 'conf_threshold':0}, #H W of cropped img
                {  'top_k': 300, "res":(128,256), 'conf_threshold':0}, #H W of cropped img
    ]



    for params in eval_params:
        hp_dataset = PatchesDataset_jj(root_dir=args.input_dir, use_color=True, output_shape=params['res'],
                                type='a')
        data_loader = DataLoader(hp_dataset,
                                 batch_size=1,
                                 pin_memory=False,
                                 shuffle=False,
                                 num_workers=8,
                                 worker_init_fn=None,
                                 sampler=None)

        print(colored('Evaluating for -- top_k {} conf_trd {}'.format( params['top_k'],params['conf_threshold']),'green'))
        rep, loc, c1, c3, c5, mscore = evaluate_keypoint_net_jj(
            data_loader,
            keypoint_net,
            params['res'],
            params['top_k'],
            params['conf_threshold'],
            use_color=True)
        #(data_loader, keypoint_net, output_shape, top_k, conf_threshold, use_color=True)

        print('Repeatability {0:.3f}'.format(rep))
        print('Localization Error {0:.3f}'.format(loc))
        print('Correctness d1 {:.3f}'.format(c1))
        print('Correctness d3 {:.3f}'.format(c3))
        print('Correctness d5 {:.3f}'.format(c5))
        print('MScore {:.3f}'.format(mscore))


def main2():
    parser = argparse.ArgumentParser(
        description='Script for KeyPointNet testing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pretrained_model", type=str, help="pretrained model path",
                        default="/home/jinjing/Projects/keypoints_comparision/git_src_code/kp2d/pretrained_models/v4.ckpt")
    parser.add_argument("--input_dir", type=str, help="Folder containing input images",
                        default="/home/jinjing/Projects/data_old/new_data/output/idx_video_frame/")#"/home/jinjing/Projects/data/out_imgs/")#/home/jinjing/Projects/data_old/new_data/output/idx_video_frame

    args = parser.parse_args()
    checkpoint = torch.load(args.pretrained_model)
    model_args = checkpoint['config']['model']['params']

    # Create and load disp net
    keypoint_net = KeypointNet(use_color=model_args['use_color'],
                               do_upsample=model_args['do_upsample'],
                               do_cross=model_args['do_cross'])
    keypoint_net.load_state_dict(checkpoint['state_dict'])
    keypoint_net = keypoint_net.cuda()
    keypoint_net.eval()
    print('Loaded KeypointNet from {}'.format(args.pretrained_model))
    print('KeypointNet params {}'.format(model_args))

    img_size = (240,320)  #(128,256)

    eval_params = [
                {  'top_k': 100, "res":img_size, 'conf_threshold':0},
                {  'top_k': 200, "res":img_size, 'conf_threshold':0}, #H W of cropped img
                {  'top_k': 300, "res":img_size, 'conf_threshold':0}, #H W of cropped img
    ]




    for params in eval_params:
        hp_dataset = PatchesDataset_jj2(root_dir=args.input_dir, use_color=True, output_shape=params['res'],
                                type='a')
        data_loader = DataLoader(hp_dataset,
                                 batch_size=1,
                                 pin_memory=False,
                                 shuffle=False,
                                 num_workers=8,
                                 worker_init_fn=None,
                                 sampler=None)

        print(colored('Evaluating for -- top_k {} conf_trd {}'.format( params['top_k'],params['conf_threshold']),'green'))
        rep, loc, c1, c3, c5, mscore = evaluate_keypoint_net_jj2(
            data_loader,
            keypoint_net,
            params['res'],
            params['top_k'],
            params['conf_threshold'],
            use_color=True)
        #(data_loader, keypoint_net, output_shape, top_k, conf_threshold, use_color=True)




        print('Repeatability {0:.3f}'.format(rep))
        print('Localization Error {0:.3f}'.format(loc))
        print('Correctness d1 {:.3f}'.format(c1))
        print('Correctness d3 {:.3f}'.format(c3))
        print('Correctness d5 {:.3f}'.format(c5))
        print('MScore {:.3f}'.format(mscore))


if __name__ == '__main__':
    # main()  # this function is for man made version images
    main2()   # this function is for temporal successive actual transformation + triangulation
