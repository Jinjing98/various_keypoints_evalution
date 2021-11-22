# Copyright 2020 Toyota Research Institute.  All rights reserved.
# Example usage: python scripts/eval_keypoint_net.sh --pretrained_model /data/models/kp2d/v4.pth --input_dir /data/datasets/kp2d/HPatches/
import cv2
import time
import pathlib
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from termcolor import colored
import cv2
import numpy as np
# import tensorflow as tf  # noqa: E402
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


from superpoint.evaluations.evaluate import evaluate_keypoint_net_SP,evaluate_keypoint_net_SP2

from match_features_demo import  extract_superpoint_keypoints_and_descriptors, preprocess_image


#comment below line of code, the code will run in GPU mode
# # this line of code will let the code to run on CPU, but this have to be in the very beginning of the code
tf.config.set_visible_devices([], 'GPU')
#
#
#below line of code will display the log for debug to see if it is real GPU
# To find out which devices your operations and tensors are assigned to
# tf.debugging.set_log_device_placement(True)


# below code may be needed if we want the code to run on GPU correctly.
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            #below is an important line of code for enabling gpu
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs are: ",gpus, "  ",len(logical_gpus), "Logical GPUs are :",logical_gpus)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)




def preprocess_image(img_file):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)


    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, 2)
    img = img.astype(np.float32)
    img_preprocessed = img / 255.

    return img_preprocessed



def main():


    ori_img_dir = "/home/jinjing/Projects/data/ori_imgs/"
    # ori_img_paths = os.listdir(ori_img_dir)

    warp_img_dir = "/home/jinjing/Projects/data/out_imgs/"
    final_ori_img_dir = "/home/jinjing/Projects/data/final_ori_imgs/"
    EXPER_PATH="/home/jinjing/Projects/keypoints_comparision/pretrained_models"

    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Compute the homography \
            between two images with the SuperPoint feature matches.')
    parser.add_argument('--k_best', type=int, default=200,
                        help='Maximum number of keypoints to keep \
                        (default: 1000)')
    args = parser.parse_args()

    weights_name = "sp_v6"#args.weights_name""
    keep_k_best = args.k_best

    weights_root_dir = Path(EXPER_PATH, 'saved_models')
    weights_root_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path(weights_root_dir, weights_name)

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        tf.saved_model.loader.load(sess,
                                   [tf.saved_model.tag_constants.SERVING],#r"E:\Google Drive\\files.sem3\NCT\Reuben_lab\keypoint_detector_descriptor_evaluator-main\models\SuperPoint\pretrained_models\sp_v6.tar")
                                   str(weights_dir))

        input_img_tensor = graph.get_tensor_by_name('superpoint/image:0')
        output_prob_nms_tensor = graph.get_tensor_by_name(
            'superpoint/prob_nms:0')
        output_desc_tensors = graph.get_tensor_by_name(
            'superpoint/descriptors:0')


        conf_trd_sets = [0]
        top_k_points = [50,100,200,300]
        for confidence in conf_trd_sets:
            for k in top_k_points:

                rep, loc, c1, c3, c5, mscore = evaluate_keypoint_net_SP(final_ori_img_dir,warp_img_dir,sess,k,input_img_tensor,
                                         output_prob_nms_tensor,output_desc_tensors,confidence )
                print(colored('Evaluating for super point: confidence {} k_points {}'.format(confidence,k),'green'))
                print('Repeatability {0:.3f}'.format(rep))
                print('Localization Error {0:.3f}'.format(loc))
                print('Correctness d1 {:.3f}'.format(c1))
                print('Correctness d3 {:.3f}'.format(c3))
                print('Correctness d5 {:.3f}'.format(c5))
                print('MScore {:.3f}'.format(mscore))



def main2():


    ori_img_dir = "/home/jinjing/Projects/data/ori_imgs/"
    # ori_img_paths = os.listdir(ori_img_dir)

    warp_img_dir = "/home/jinjing/Projects/data/out_imgs/"
    final_ori_img_dir = "/home/jinjing/Projects/data/final_ori_imgs/"
    warp_img_dir = "/home/jinjing/Projects/data_old/new_data/output/paired_opt/"#idx_video_frame
    final_ori_img_dir = "/home/jinjing/Projects/data_old/new_data/output/idx_video_frame/"

    EXPER_PATH="/home/jinjing/Projects/keypoints_comparision/pretrained_models"

    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Compute the homography \
            between two images with the SuperPoint feature matches.')
    parser.add_argument('--k_best', type=int, default=200,
                        help='Maximum number of keypoints to keep \
                        (default: 1000)')
    args = parser.parse_args()

    weights_name = "sp_v6"#args.weights_name""
    keep_k_best = args.k_best

    weights_root_dir = Path(EXPER_PATH, 'saved_models')
    weights_root_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path(weights_root_dir, weights_name)

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        tf.saved_model.loader.load(sess,
                                   [tf.saved_model.tag_constants.SERVING],#r"E:\Google Drive\\files.sem3\NCT\Reuben_lab\keypoint_detector_descriptor_evaluator-main\models\SuperPoint\pretrained_models\sp_v6.tar")
                                   str(weights_dir))

        input_img_tensor = graph.get_tensor_by_name('superpoint/image:0')
        output_prob_nms_tensor = graph.get_tensor_by_name(
            'superpoint/prob_nms:0')
        output_desc_tensors = graph.get_tensor_by_name(
            'superpoint/descriptors:0')


        conf_trd_sets = [0]
        top_k_points = [50,100,200,300]
        for confidence in conf_trd_sets:
            for k in top_k_points:

                rep, loc, c1, c3, c5, mscore = evaluate_keypoint_net_SP2(final_ori_img_dir,warp_img_dir,sess,k,input_img_tensor,
                                         output_prob_nms_tensor,output_desc_tensors,confidence )
                print(colored('Evaluating for super point: confidence {} k_points {}'.format(confidence,k),'green'))
                print('Repeatability {0:.3f}'.format(rep))
                print('Localization Error {0:.3f}'.format(loc))
                print('Correctness d1 {:.3f}'.format(c1))
                print('Correctness d3 {:.3f}'.format(c3))
                print('Correctness d5 {:.3f}'.format(c5))
                print('MScore {:.3f}'.format(mscore))



if __name__ == '__main__':
    # main()
    main2()































#
#
#
# import argparse
# import os
# import pickle
# import random
# import subprocess
#
# import cv2
# import numpy as np
# import torch
# from PIL import Image
# from termcolor import colored
# from torch.utils.data import DataLoader, Dataset
# from tqdm import tqdm
# import sys
#
#
# sys.path.append('..')  #  this is important for same level import
# # #https://blog.csdn.net/songbinxu/article/details/80289489
# # #https://cloud.tencent.com/developer/article/1803921
# # #https://www.jianshu.com/p/5a02285bb111
#
# from kp2d.datasets.patches_dataset import PatchesDataset_jj #PatchesDataset
# from  kp2d.evaluation.evaluate import evaluate_keypoint_net_jj#evaluate_keypoint_net
# from  kp2d.networks.keypoint_net import KeypointNet
#
#
# def main():
#     parser = argparse.ArgumentParser(
#         description='Script for KeyPointNet testing',
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument("--pretrained_model", type=str, help="pretrained model path",
#                         default="/home/jinjing/Projects/keypoints_comparision/git_src_code/kp2d/pretrained_models/v4.ckpt")
#     parser.add_argument("--input_dir", type=str, help="Folder containing input images",
#                         default="/home/jinjing/Projects/data/out_imgs/")
#
#     args = parser.parse_args()
#     checkpoint = torch.load(args.pretrained_model)
#     model_args = checkpoint['config']['model']['params']
#
#     # Create and load disp net
#     keypoint_net = KeypointNet(use_color=model_args['use_color'],
#                                do_upsample=model_args['do_upsample'],
#                                do_cross=model_args['do_cross'])
#     keypoint_net.load_state_dict(checkpoint['state_dict'])
#     keypoint_net = keypoint_net.cuda()
#     keypoint_net.eval()
#     print('Loaded KeypointNet from {}'.format(args.pretrained_model))
#     print('KeypointNet params {}'.format(model_args))
#
#     eval_params = [
#                 # {  'top_k': 100, },
#                 # {  'top_k': 200, },
#                 {  'top_k': 300, },
#
#     ]
#
#     for params in eval_params:
#         hp_dataset = PatchesDataset_jj(root_dir=args.input_dir, use_color=True,   # output_shape=params['res'],
#                                 type='a')
#         data_loader = DataLoader(hp_dataset,
#                                  batch_size=1,
#                                  pin_memory=False,
#                                  shuffle=False,
#                                  num_workers=8,
#                                  worker_init_fn=None,
#                                  sampler=None)
#
#         print(colored('Evaluating for -- top_k {}'.format( params['top_k']),'green'))
#         rep, loc, c1, c3, c5, mscore = evaluate_keypoint_net_jj(
#             data_loader,
#             keypoint_net,
#             # output_shape=params['res'],
#
#             top_k=params['top_k'],
#             use_color=True)
#
#         print('Repeatability {0:.3f}'.format(rep))
#         print('Localization Error {0:.3f}'.format(loc))
#         print('Correctness d1 {:.3f}'.format(c1))
#         print('Correctness d3 {:.3f}'.format(c3))
#         print('Correctness d5 {:.3f}'.format(c5))
#         print('MScore {:.3f}'.format(mscore))


# if __name__ == '__main__':
#     main()
