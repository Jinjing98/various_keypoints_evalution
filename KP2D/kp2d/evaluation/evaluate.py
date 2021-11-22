# Copyright 2020 Toyota Research Institute.  All rights reserved.

import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
# import sys,os
# sys.path.append('.')
# rom


import os, sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
from descriptor_evaluation import *
from detector_evaluation import *
from ..utils.image import to_color_normalized,to_gray_normalized

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


# from  descriptor_evaluation import compute_homography, compute_matching_score
# from  detector_evaluation import compute_repeatability

# from  utils.image import to_color_normalized, to_gray_normalized


def evaluate_keypoint_net_jj(data_loader, keypoint_net, output_shape, top_k, conf_threshold, use_color=True):
    """Keypoint net evaluation script.

    Parameters
    ----------
    data_loader: torch.utils.data.DataLoader
        Dataset loader.
    keypoint_net: torch.nn.module
        Keypoint network.
    output_shape: tuple
        Original image shape.
    top_k: int
        Number of keypoints to use to compute metrics, selected based on probability.
    use_color: bool
        Use color or grayscale images.
    """
    keypoint_net.eval()
    keypoint_net.training = False


    localization_err, repeatability = [], []
    correctness1, correctness3, correctness5, MScore = [], [], [], []


    with torch.no_grad():
        for i, sample in tqdm(enumerate(data_loader), desc="evaluate_keypoint_net"):
            # a = sample['image'] # 1 3 128 256       128 256 3
            # print("kk:",a.shape,a)
            if use_color:
                image = to_color_normalized(sample['image'].cuda())
                # print("debug:",image.size())
                warped_image = to_color_normalized(sample['warped_image'].cuda())
            else:
                image = to_gray_normalized(sample['image'].cuda())
                warped_image = to_gray_normalized(sample['warped_image'].cuda())

            score_1, coord_1, desc1 = keypoint_net(image)
            score_2, coord_2, desc2 = keypoint_net(warped_image)
            B, _, Hc, Wc = desc1.shape

            # Scores & Descriptors
            score_1 = torch.cat([coord_1, score_1], dim=1).view(3, -1).t().cpu().numpy()
            score_2 = torch.cat([coord_2, score_2], dim=1).view(3, -1).t().cpu().numpy()
            desc1 = desc1.view(256, Hc, Wc).view(256, -1).t().cpu().numpy()
            desc2 = desc2.view(256, Hc, Wc).view(256, -1).t().cpu().numpy()
            # print("debug",score_1.shape,score_2.shape)  #(310.220.prob)  (w,h.prob)


            # Filter based on confidence threshold
            desc1 = desc1[score_1[:, 2] > conf_threshold, :]
            desc2 = desc2[score_2[:, 2] > conf_threshold, :]
            score_1 = score_1[score_1[:, 2] > conf_threshold, :]
            score_2 = score_2[score_2[:, 2] > conf_threshold, :]

            # Prepare data for eval
            data = {'image': sample['image'].numpy().squeeze(),
                    'image_shape' : output_shape[::-1],
                    'warped_image': sample['warped_image'].numpy().squeeze(),
                    'homography': sample['homography'].squeeze().numpy(),
                    'prob': score_1,
                    'warped_prob': score_2,
                    'desc': desc1,
                    'warped_desc': desc2}

            # Compute repeatabilty and localization error
            # print("the img: ",i)
            _, _, rep, loc_err = compute_repeatability(data, keep_k_points=top_k, distance_thresh=3)
            repeatability.append(rep)
            localization_err.append(loc_err)

            # Compute correctness
            c1, c2, c3 = compute_homography(data, keep_k_points=top_k)
            correctness1.append(c1)
            correctness3.append(c2)
            correctness5.append(c3)

            # Compute matching score
            mscore = compute_matching_score(data, keep_k_points=top_k)
            MScore.append(mscore)

    return np.mean(repeatability), np.mean(localization_err), \
           np.mean(correctness1), np.mean(correctness3), np.mean(correctness5), np.mean(MScore)


def evaluate_keypoint_net_jj2(data_loader, keypoint_net, output_shape, top_k, conf_threshold, use_color=True):
    """Keypoint net evaluation script.

    Parameters
    ----------
    data_loader: torch.utils.data.DataLoader
        Dataset loader.
    keypoint_net: torch.nn.module
        Keypoint network.
    output_shape: tuple
        Original image shape.
    top_k: int
        Number of keypoints to use to compute metrics, selected based on probability.
    use_color: bool
        Use color or grayscale images.
    """
    keypoint_net.eval()
    keypoint_net.training = False


    localization_err, repeatability = [], []
    correctness1, correctness3, correctness5, MScore = [], [], [], []




    with torch.no_grad():
        for i, sample in tqdm(enumerate(data_loader), desc="evaluate_keypoint_net"):
            if use_color:
                image = to_color_normalized(sample['image'].cuda())
                # print("debug:",image.size())
                warped_image = to_color_normalized(sample['warped_image'].cuda())
                pts1 = sample['pts1']
                pts2 = sample['pts2']



            else:
                image = to_gray_normalized(sample['image'].cuda())
                warped_image = to_gray_normalized(sample['warped_image'].cuda())

            score_1, coord_1, desc1 = keypoint_net(image)
            score_2, coord_2, desc2 = keypoint_net(warped_image)
            B, _, Hc, Wc = desc1.shape
            # print(score_1.shape)  # 1 256 30 40 //// 1 1 30 40   #height/8  weight/8

            # Scores & Descriptors
            score_1 = torch.cat([coord_1, score_1], dim=1).view(3, -1).t().cpu().numpy()
            score_2 = torch.cat([coord_2, score_2], dim=1).view(3, -1).t().cpu().numpy()
            desc1 = desc1.view(256, Hc, Wc).view(256, -1).t().cpu().numpy()
            desc2 = desc2.view(256, Hc, Wc).view(256, -1).t().cpu().numpy()
            # print(desc1.shape,score_1.shape)  # (1200, 256) (1200, 3)
            w1,h1,w2,h2,w3,h3,w4,h4 = list(pts1.reshape((8,)))
            w1_2,h1_2,w2_2,h2_2,w3_2,h3_2,w4_2,h4_2 = list(pts2.reshape((8,)))
            # print("k:",list(pts1.reshape((6,))))
            polygon1 = Polygon([(w1, h1), (w2, h2), (w3, h3),(w4,h4)])
            polygon2 = Polygon([(w1_2, h1_2), (w2_2, h2_2), (w3_2, h3_2),(w4_2,h4_2)])

            img = np.moveaxis(np.array(sample["image"][0].numpy()).astype(np.uint8), 0, -1)
            warpped_img = np.moveaxis(np.array(sample["warped_image"][0].numpy()).astype(np.uint8), 0, -1)

# below code will filltered out the kps out of triangle
            for idx,i in enumerate(score_1):
                point1 = Point(i[0],i[1])  # x y (w,h)
                # print("before:",score_1[idx][2])
                if polygon1.contains(point1)==False:
                    score_1[idx][2] = -10000
                    continue
                img = cv2.circle(img,(i[0],i[1]),radius=0, color=(0, 0, 100), thickness=2)


            for idx,i in enumerate(score_2):
                point2 = Point(i[0],i[1])  # x y (w,h)
                # print("before:",score_1[idx][2])
                if polygon2.contains(point2)==False:
                    score_2[idx][2] = -10000
                    continue
                warpped_img = cv2.circle(warpped_img,(i[0],i[1]),radius=0, color=(0, 0, 200), thickness=2)

            #  read the pts1 pts2 from the files then vis pts within the trian
            img = cv2.circle(img, (w1,h1), radius=0, color=(0, 0, 255), thickness=5)
            img = cv2.circle(img, (w2,h2), radius=0, color=(0, 0, 255), thickness=5)
            img = cv2.circle(img, (w3,h3), radius=0, color=(0, 0, 255), thickness=5)
            img = cv2.circle(img, (w4,h4), radius=0, color=(0, 0, 255), thickness=5)
            warpped_img = cv2.circle(warpped_img, (w1_2,h1_2), radius=0, color=(0, 0, 255), thickness=5)
            warpped_img = cv2.circle(warpped_img, (w2_2,h2_2), radius=0, color=(0, 0, 255), thickness=5)
            warpped_img = cv2.circle(warpped_img, (w3_2,h3_2), radius=0, color=(0, 0, 255), thickness=5)
            warpped_img = cv2.circle(warpped_img, (w4_2,h4_2), radius=0, color=(0, 0, 255), thickness=5)
            # cv2.imshow("",np.concatenate((img.get(),warpped_img.get())))
            # cv2.waitKey(0)






            # conf_threshold = -10000
            # Filter based on confidence threshold
            # print("debug1:",score_1.shape,score_2.shape)#1024 3  #3:(w,h.prob)
            desc1 = desc1[score_1[:, 2] > conf_threshold, :]
            desc2 = desc2[score_2[:, 2] > conf_threshold, :]
            score_1 = score_1[score_1[:, 2] > conf_threshold, :]  # w h p  #1064  3
            score_2 = score_2[score_2[:, 2] > conf_threshold, :]
            # print("debug2:",score_1.shape,score_2.shape)#1024 3  #3:(w,h.prob)

            # Prepare data for eval
            data = {'image': sample['image'].numpy().squeeze(),
                    'image_shape' : output_shape[::-1],
                    'warped_image': sample['warped_image'].numpy().squeeze(),
                    'homography': sample['homography'].squeeze().numpy(),  # GT homo mat
                    'prob': score_1,
                    'warped_prob': score_2,
                    'desc': desc1,
                    'warped_desc': desc2}

            # Compute repeatabilty and localization error
            # print("the img: ",i)
            _, _, rep, loc_err = compute_repeatability(data, keep_k_points=top_k, distance_thresh=3)
            repeatability.append(rep)
            localization_err.append(loc_err)
            #
            # Compute correctness
            c1, c2, c3 = compute_homography(data, keep_k_points=top_k)
            correctness1.append(c1)
            correctness3.append(c2)
            correctness5.append(c3)

            #
            # Compute matching score
            mscore = compute_matching_score(data, keep_k_points=top_k)
            MScore.append(mscore)
            # print("loc error is : ",localization_err,correctness1)

    return np.mean(repeatability), np.mean(localization_err), \
           np.mean(correctness1), np.mean(correctness3), np.mean(correctness5), np.mean(MScore)

