# Copyright 2020 Toyota Research Institute.  All rights reserved.
# Adapted from: https://github.com/rpautrat/SuperPoint/blob/master/superpoint/evaluations/detector_evaluation.py

import random
from glob import glob
from os import path as osp

import cv2
import numpy as np
# from brand_new.KP2D.kp2d.utils.keypoints import warp_keypoints



def warp_keypoints(keypoints, H):
    """Warp keypoints given a homography

    Parameters
    ----------
    keypoints: numpy.ndarray (N,2)
        Keypoint vector.
    H: numpy.ndarray (3,3)
        Homography.

    Returns
    -------
    warped_keypoints: numpy.ndarray (N,2)
        Warped keypoints vector.
    """
    num_points = keypoints.shape[0]
    homogeneous_points = np.concatenate([keypoints, np.ones((num_points, 1))], axis=1)
    warped_points = np.dot(homogeneous_points, np.transpose(H))  #
    return warped_points[:, :2] / warped_points[:, 2:]


    # keypoints =
    # warped_keypoints =
    # prob =
    # warped_prob =
    # H =

def compute_repeatability(keypoints,warped_keypoints,H,shape,distance_thresh):
    """
    Compute the repeatability metric between 2 sets of keypoints inside data.

    Parameters
    ----------
    data: dict
        Input dictionary containing:
        image_shape: tuple (H,W)
            Original image shape.
        homography: numpy.ndarray (3,3)
            Ground truth homography.
        prob: numpy.ndarray (N,3)
            Keypoint vector, consisting of (x,y,probability).
        warped_prob: numpy.ndarray (N,3)
            Warped keypoint vector, consisting of (x,y,probability).
    keep_k_points: int
        Number of keypoints to select, based on probability.
    distance_thresh: int
        Distance threshold in pixels for a corresponding keypoint to be considered a correct match.

    Returns
    -------    
    N1: int
        Number of true keypoints in the first image.
    N2: int
        Number of true keypoints in the second image.
    repeatability: float
        Keypoint repeatability metric.
    loc_err: float
        Keypoint localization error.
    """

    #shape 128 256 1

    def filter_keypoints(points, shape):
        """ Keep only the points whose coordinates are inside the dimensions of shape. """
        mask = (points[:, 0] >= 0) & (points[:, 0] < shape[0]) &\
               (points[:, 1] >= 0) & (points[:, 1] < shape[1])
        return points[mask, :]

    def keep_true_keypoints(points, H, shape):
        """ Keep only the points whose warped coordinates by H are still inside shape. """
        # a = points[:, [1, 0]]
        warped_points = warp_keypoints(points[:, [1, 0]], H)
        warped_points[:, [0, 1]] = warped_points[:, [1, 0]]
        mask = (warped_points[:, 0] >= 0) & (warped_points[:, 0] < shape[0]) &\
               (warped_points[:, 1] >= 0) & (warped_points[:, 1] < shape[1])
        return points[mask, :]

    width = shape[1]
    height = shape[0]
    keypoints_data = [[p.pt[1],p.pt[0]] for p in keypoints]  #y x  #jj
    warped_keypoints_data = [[p.pt[1],p.pt[0]] for p in warped_keypoints]
    keypoints = np.array(keypoints_data).reshape(-1,2)
    warped_keypoints = np.array(warped_keypoints_data).reshape(-1,2)

    #
    # keypoints = np.stack([keypoints[0], keypoints[1]], axis=-1)
    # warped_keypoints = np.stack([warped_keypoints[0], warped_keypoints[1]], axis=-1)



    # warped_keypoints = np.stack([warped_keypoints[0], warped_keypoints[1], warped_prob], axis=-1)
    warped_keypoints = keep_true_keypoints(warped_keypoints, np.linalg.inv(H), shape)


    # Warp the original keypoints with the true homography
    true_warped_keypoints = warp_keypoints(keypoints[:, [1, 0]], H)
    true_warped_keypoints = np.stack([true_warped_keypoints[:, 1], true_warped_keypoints[:, 0]], axis=-1)

    # true_warped_keypoints = np.stack([true_warped_keypoints[:, 1], true_warped_keypoints[:, 0], prob], axis=-1)
    true_warped_keypoints = filter_keypoints(true_warped_keypoints, shape)
    #
    # # Keep only the keep_k_points best predictions
    # warped_keypoints = select_k_best(warped_keypoints, keep_k_points)
    # true_warped_keypoints = select_k_best(true_warped_keypoints, keep_k_points)
    #
    #
    #





    # Compute the repeatability
    N1 = true_warped_keypoints.shape[0]
    N2 = warped_keypoints.shape[0]
    true_warped_keypoints = np.expand_dims(true_warped_keypoints, 1)
    warped_keypoints = np.expand_dims(warped_keypoints, 0)
    # shapes are broadcasted to N1 x N2 x 2:
    norm = np.linalg.norm(true_warped_keypoints - warped_keypoints, ord=None, axis=2)
    count1 = 0
    count2 = 0
    le1 = 0
    le2 = 0
    if N2 != 0:
        min1 = np.min(norm, axis=1)
        correct1 = (min1 <= distance_thresh)
        count1 = np.sum(correct1)
        le1 = min1[correct1].sum()
    if N1 != 0:
        min2 = np.min(norm, axis=0)
        correct2 = (min2 <= distance_thresh)
        count2 = np.sum(correct2)
        le2 = min2[correct2].sum()
    if N1 + N2 > 0:
        repeatability = (count1 + count2) / (N1 + N2)
        loc_err = (le1 + le2) / (count1 + count2)
    else:
        repeatability = -1
        loc_err = -1

    return N1, N2, repeatability, loc_err
