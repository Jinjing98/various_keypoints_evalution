import argparse
from pathlib import Path

import cv2
import numpy as np
# import tensorflow as tf  # noqa: E402
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# from settings import EXPER_PATH  # noqa: E402
EXPER_PATH=r'E:\Google Drive\\files.sem3\NCT\Reuben_lab\keypoint_detector_descriptor_evaluator-main\models\SuperPoint\pretrained_models'

def extract_SIFT_keypoints_and_descriptors(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(np.squeeze(gray_img), None)

    return kp, desc

def extract_superpoint_keypoints_and_descriptors(keypoint_map, descriptor_map,
                                                 keep_k_points=1000):

    def select_k_best(points, k):
        """ Select the k most probable points (and strip their proba).
        points has shape (num_points, 3) where the last coordinate is the proba. """
        sorted_prob = points[points[:, 2].argsort(), :2]
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :]

    # Extract keypoints
    # confidence set to zero?
    confidence = 0
    keypoints = np.where(keypoint_map > confidence)
    prob = keypoint_map[keypoints[0], keypoints[1]]
    keypoints = np.stack([keypoints[0], keypoints[1], prob], axis=-1)

    keypoints = select_k_best(keypoints, keep_k_points)
    keypoints = keypoints.astype(int)

    # Get descriptors for keypoints
    desc = descriptor_map[keypoints[:, 0], keypoints[:, 1]]

    # Convert from just pts to cv2.KeyPoints
    keypoints = [cv2.KeyPoint(p[1], p[0], 1) for p in keypoints]

    return keypoints, desc


def match_descriptors(kp1, desc1, kp2, desc2):
    # Match the keypoints with the warped_keypoints with nearest neighbor search
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches_idx = np.array([m.queryIdx for m in matches])
    m_kp1 = [kp1[idx] for idx in matches_idx]
    matches_idx = np.array([m.trainIdx for m in matches])
    m_kp2 = [kp2[idx] for idx in matches_idx]

    return m_kp1, m_kp2, matches


def compute_homography(matched_kp1, matched_kp2):
    matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
    matched_pts2 = cv2.KeyPoint_convert(matched_kp2)

    # Estimate the homography between the matches using RANSAC
    H, inliers = cv2.findHomography(matched_pts1[:, [1, 0]],
                                    matched_pts2[:, [1, 0]],
                                    cv2.RANSAC)
    inliers = inliers.flatten()
    return H, inliers


def preprocess_image(img_file):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)


    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, 2)
    img = img.astype(np.float32)
    img_preprocessed = img / 255.

    return img_preprocessed

