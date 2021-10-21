# Copyright 2020 Toyota Research Institute.  All rights reserved.

# import numpy as np
# import torch
# import torchvision.transforms as transforms
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
# import sys,os
# sys.path.append('.')
# # rom
#
#
import os, sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
# from descriptor_evaluation import *
# from detector_evaluation import *
# from ..utils.image import to_color_normalized,to_gray_normalized

from detector_evaluation import *
from descriptor_evaluation import *

def preprocess_image(img_file):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)


    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, 2)
    img = img.astype(np.float32)
    img_preprocessed = img / 255.

    return img_preprocessed



def to_gray_normalized(images):
    """Performs image normalization and converts images to grayscale (preserving dimensions)

    Parameters
    ----------
    images: torch.Tensor
        Input images.

    Returns
    -------
    normalized_images: torch.Tensor
        Normalized grayscale images.
    """
    assert len(images.shape) == 4
    images -= 0.5
    images *= 0.225
    normalized_images = images.mean(1).unsqueeze(1)
    return normalized_images


def to_color_normalized(images):
    """Performs image normalization and converts images to grayscale (preserving dimensions)

    Parameters
    ----------
    images: torch.Tensor
        Input images.

    Returns
    -------
    normalized_images: torch.Tensor
        Normalized grayscale images.
    """
    assert len(images.shape) == 4
    images -= 0.5
    images *= 0.225
    return images



def extract_superpoint_keypoints_and_descriptors(keypoint_map, descriptor_map,keep_k_points,
                                                 confidence):

    def select_k_best(points, k):
        """ Select the k most probable points (and strip their proba).
        points has shape (num_points, 3) where the last coordinate is the proba. """
        sorted_prob = points[points[:, 2].argsort(), :2]
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :]

    # Extract keypoints
    # confidence set to zero?
    # confidence = 0
    # Extract keypoints
    ids = np.where(keypoint_map > confidence)#128*256->tuple:2 (row_id),(col_id)
    prob = keypoint_map[ids[0], ids[1]]  # (199,)
    id_prob = np.stack([ids[0], ids[1], prob], axis=-1)#(199,3)

    keypoints = select_k_best(id_prob, keep_k_points)
    keypoints = keypoints.astype(int)






    # Get descriptors for keypoints
    desc = descriptor_map[keypoints[:, 0], keypoints[:, 1]]

    # Convert from just pts to cv2.KeyPoints
    keypoints = [cv2.KeyPoint(p[1], p[0], 1) for p in keypoints]

    return keypoints, desc



def evaluate_keypoint_net_SP(ori_dir,warp_dir,sess,top_k,input_img_tensor,output_prob_nms_tensor,output_desc_tensors,conf_threshold ):
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
    # keypoint_net.eval()
    # keypoint_net.training = False


    localization_err, repeatability = [], []
    correctness1, correctness3, correctness5, MScore = [], [], [], []


    image_paths = []
    warped_image_paths = []
    file_ext = ".png"

    folder_paths = [x for x in Path(warp_dir).iterdir() if x.is_dir()]
    for trans_dir_all in folder_paths:
        ori_name = str(trans_dir_all).split("/")[-1]
        img1 = preprocess_image(ori_dir+ori_name+file_ext)
        # pridict!
        out1 = sess.run([output_prob_nms_tensor, output_desc_tensors],
                        feed_dict={input_img_tensor: np.expand_dims(img1, 0)})



        keypoint_map1 = np.squeeze(out1[0])
        descriptor_map1 = np.squeeze(out1[1])

        kp1, des1 = extract_superpoint_keypoints_and_descriptors(
                keypoint_map1, descriptor_map1, top_k,confidence=conf_threshold)

        ori_img = cv2.imread(ori_dir+ori_name+file_ext)
        outimg = ori_img.copy()
        cv2.drawKeypoints(ori_img,kp1,outimg)
        # cv2.imshow("",outimg)
        # cv2.waitKey(0)




        for trans_dir_sig in trans_dir_all.iterdir():
            H_dir = np.load(str(trans_dir_sig)+"/GT_H_mat.npz")
            num_images = 6  # for each kind of trans we have 6
            for trans_name in [f for f in os.listdir(trans_dir_sig) if f.endswith('.png')]:
                # print("now its the ",str(trans_dir_sig)+"/"+trans_name+file_ext)
                trans_name = trans_name[:-4]
                # image_paths.append(ori_dir+ori_name+file_ext)
                # warp_path = str(trans_dir_sig)+"/"+trans_name+file_ext
                H = H_dir[trans_name]
                img2 =  preprocess_image(str(trans_dir_sig)+"/"+trans_name+file_ext)
                shape = img2.shape
                distance_thresh = 3
                out2 = sess.run([output_prob_nms_tensor, output_desc_tensors],
                                feed_dict={input_img_tensor: np.expand_dims(img2, 0)})
                keypoint_map2 = np.squeeze(out2[0])
                descriptor_map2 = np.squeeze(out2[1])
# des2 300*256
                kp2, des2 = extract_superpoint_keypoints_and_descriptors(
                        keypoint_map2, descriptor_map2, top_k,confidence=conf_threshold) # for leared feature, the conf and top k are already filtered here


                ori_img2 = cv2.imread(str(trans_dir_sig)+"/"+trans_name+file_ext)
                outimg2 = ori_img2.copy()
                cv2.drawKeypoints(ori_img2,kp2,outimg2)
                # cv2.imshow("",np.concatenate((outimg,outimg2),axis = 1))
                # cv2.waitKey(0)
                #kp1 kp2 should be fixed

                N1, N2, rep, loc_err= compute_repeatability(kp1,kp2,H,shape,distance_thresh = 3)



                repeatability.append(rep)
                localization_err.append(loc_err)

                # Compute correctness
                # c1, c2, c3 = compute_homography(data, keep_k_points=top_k)
                c1, c2, c3 = compute_homography(H,shape,kp1,kp2,des1,des2)
                correctness1.append(c1)
                correctness3.append(c2)
                correctness5.append(c3)

                # Compute matching score
                mscore = compute_matching_score(H,shape,kp1,kp2,des1,des2)
                MScore.append(mscore)

    return np.nanmean(repeatability), np.nanmean(localization_err),\
           np.nanmean(correctness1), np.nanmean(correctness3), np.nanmean(correctness5),np.nanmean(MScore)



#nfeatures=None, scaleFactor=None, nlevels=None, edgeThreshold=None,
# firstLevel=None, WTA_K=None, scoreType=None, patchSize=None, fastThreshold=None

def evaluate_keypoint_net_ORB(ori_dir,warp_dir,num4features,fast_trd):
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
    # keypoint_net.eval()
    # keypoint_net.training = False

    conf_threshold = 0
    localization_err, repeatability = [], []
    correctness1, correctness3, correctness5, MScore = [], [], [], []


    image_paths = []
    warped_image_paths = []
    file_ext = ".png"

    folder_paths = [x for x in Path(warp_dir).iterdir() if x.is_dir()]
    orb = cv2.ORB_create(nfeatures=num4features,fastThreshold=fast_trd)

    for trans_dir_all in folder_paths:
        ori_name = str(trans_dir_all).split("/")[-1]

        img_1 = cv2.imread(ori_dir+ori_name+file_ext)
        kp1, des1 = orb.detectAndCompute(img_1,None)  # maybe less than 500

        outimg = img_1.copy()
        cv2.drawKeypoints(img_1,kp1,outimg)
        # cv2.imshow("",outimg)
        # cv2.waitKey(0)




        for trans_dir_sig in trans_dir_all.iterdir():
            H_dir = np.load(str(trans_dir_sig)+"/GT_H_mat.npz")
            num_images = 6  # for each kind of trans we have 6
            for trans_name in [f for f in os.listdir(trans_dir_sig) if f.endswith('.png')]:

                trans_name = trans_name[:-4]

                H = H_dir[trans_name]
                img2 =  preprocess_image(str(trans_dir_sig)+"/"+trans_name+file_ext)
                shape = img2.shape



                img_2 = cv2.imread(str(trans_dir_sig)+"/"+trans_name+file_ext)
                kp2, des2 = orb.detectAndCompute(img_2,None)  # maybe less than 500




                # ori_img2 = cv2.imread(str(trans_dir_sig)+"/"+trans_name+file_ext)
                outimg2 = img_2.copy()
                cv2.drawKeypoints(img_2,kp2,outimg2)
                # cv2.imshow("",np.concatenate((outimg,outimg2),axis = 1))
                # cv2.waitKey(0)




                N1, N2, rep, loc_err= compute_repeatability(kp1,kp2,H,shape,distance_thresh = 3)



                repeatability.append(rep)
                localization_err.append(loc_err)

                # Compute correctness
                # c1, c2, c3 = compute_homography(data, keep_k_points=top_k)
                c1, c2, c3 = compute_homography(H,shape,kp1,kp2,des1,des2)
                correctness1.append(c1)
                correctness3.append(c2)
                correctness5.append(c3)
                #
                # Compute matching score
                mscore = compute_matching_score(H,shape,kp1,kp2,des1,des2)
                MScore.append(mscore)
    # print(MScore)

    return np.nanmean(repeatability), np.nanmean(localization_err),\
           np.nanmean(correctness1), np.nanmean(correctness3), np.nanmean(correctness5),np.nanmean(MScore)




def evaluate_keypoint_net_AKAZE(ori_dir,warp_dir,diff_type,trd):
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
    # keypoint_net.eval()
    # keypoint_net.training = False

    localization_err, repeatability = [], []
    correctness1, correctness3, correctness5, MScore = [], [], [], []

    file_ext = ".png"

    folder_paths = [x for x in Path(warp_dir).iterdir() if x.is_dir()]
    #descriptor_type=None, descriptor_size=None, descriptor_channels=None,
    # threshold=None, nOctaves=None, nOctaveLayers=None, diffusivity=None

    akaze = cv2.AKAZE_create(threshold=trd,diffusivity=diff_type)
    # akaze = cv2.AKAZE_create()



    for trans_dir_all in folder_paths:
        ori_name = str(trans_dir_all).split("/")[-1]
        img_1 = cv2.imread(ori_dir+ori_name+file_ext)
        kp1,des1 = akaze.detectAndCompute(img_1,None)
        # kp1, des1 = orb.detectAndCompute(img_1,None)  # maybe less than 500

        outimg = img_1.copy()
        cv2.drawKeypoints(img_1,kp1,outimg)
        # cv2.imshow("",outimg)
        # cv2.waitKey(0)




        for trans_dir_sig in trans_dir_all.iterdir():
            H_dir = np.load(str(trans_dir_sig)+"/GT_H_mat.npz")
            for trans_name in [f for f in os.listdir(trans_dir_sig) if f.endswith('.png')]:
                trans_name = trans_name[:-4]

                H = H_dir[trans_name]
                img2 =  preprocess_image(str(trans_dir_sig)+"/"+trans_name+file_ext)
                shape = img2.shape



                img_2 = cv2.imread(str(trans_dir_sig)+"/"+trans_name+file_ext)
                kp2, des2 = akaze.detectAndCompute(img_2,None)  # maybe less than 500
                # kp2, des2 = orb.detectAndCompute(img_2,None)  # maybe less than 500




                # ori_img2 = cv2.imread(str(trans_dir_sig)+"/"+trans_name+file_ext)
                outimg2 = img_2.copy()
                cv2.drawKeypoints(img_2,kp2,outimg2)
                # cv2.imshow("",np.concatenate((outimg,outimg2),axis = 1))
                # cv2.waitKey(0)




                N1, N2, rep, loc_err= compute_repeatability(kp1,kp2,H,shape,distance_thresh = 3)



                repeatability.append(rep)
                localization_err.append(loc_err)

                c1, c2, c3 = compute_homography(H,shape,kp1,kp2,des1,des2)
                correctness1.append(c1)
                correctness3.append(c2)
                correctness5.append(c3)
                #
                # Compute matching score
                mscore = compute_matching_score(H,shape,kp1,kp2,des1,des2)
                MScore.append(mscore)


    return np.nanmean(repeatability), np.nanmean(localization_err),\
           np.nanmean(correctness1), np.nanmean(correctness3), np.nanmean(correctness5),np.nanmean(MScore)



def evaluate_keypoint_net_agast_SIFT(ori_dir,warp_dir,trd,agast_type):
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
    # keypoint_net.eval()
    # keypoint_net.training = False

    localization_err, repeatability = [], []
    correctness1, correctness3, correctness5, MScore = [], [], [], []

    file_ext = ".png"

    folder_paths = [x for x in Path(warp_dir).iterdir() if x.is_dir()]
    # akaze = cv2.AKAZE_create()
    sift = cv2.xfeatures2d.SIFT_create()
    AGAST_TYPES = {
                    '5_8': cv2.AgastFeatureDetector_AGAST_5_8,
                    'OAST_9_16': cv2.AgastFeatureDetector_OAST_9_16,
                    '7_12_d': cv2.AgastFeatureDetector_AGAST_7_12d,
                    '7_12_s': cv2.AgastFeatureDetector_AGAST_7_12s }

    agast = cv2.AgastFeatureDetector_create(threshold=trd, nonmaxSuppression=True, type=AGAST_TYPES[agast_type])






    for trans_dir_all in folder_paths:
        ori_name = str(trans_dir_all).split("/")[-1]
        img_1 = cv2.imread(ori_dir+ori_name+file_ext)
        # kp1,des1 = akaze.detectAndCompute(img_1,None)
        # kp1, des1 = orb.detectAndCompute(img_1,None)  # maybe less than 500


        kp1 = agast.detect(img_1)  #  how to restrict the num of detected kps
        kp1,des1 = sift.compute(img_1,kp1)




        outimg = img_1.copy()
        cv2.drawKeypoints(img_1,kp1,outimg)
        # cv2.imshow("",outimg)
        # cv2.waitKey(0)




        for trans_dir_sig in trans_dir_all.iterdir():
            H_dir = np.load(str(trans_dir_sig)+"/GT_H_mat.npz")
            for trans_name in [f for f in os.listdir(trans_dir_sig) if f.endswith('.png')]:
                trans_name = trans_name[:-4]

                H = H_dir[trans_name]
                img2 =  preprocess_image(str(trans_dir_sig)+"/"+trans_name+file_ext)
                shape = img2.shape



                img_2 = cv2.imread(str(trans_dir_sig)+"/"+trans_name+file_ext)
                # kp2, des2 = akaze.detectAndCompute(img_2,None)  # maybe less than 500
                # kp2, des2 = orb.detectAndCompute(img_2,None)  # maybe less than 500

                kp2 = agast.detect(img_2)  #  how to restrict the num of detected kps
                kp2,des2 = sift.compute(img_2,kp2)




                # ori_img2 = cv2.imread(str(trans_dir_sig)+"/"+trans_name+file_ext)
                outimg2 = img_2.copy()
                cv2.drawKeypoints(img_2,kp2,outimg2)
                # cv2.imshow("",np.concatenate((outimg,outimg2),axis = 1))
                # cv2.waitKey(0)




                N1, N2, rep, loc_err= compute_repeatability(kp1,kp2,H,shape,distance_thresh = 3)




                repeatability.append(rep)
                localization_err.append(loc_err)

                c1, c2, c3 = compute_homography(H,shape,kp1,kp2,des1,des2)
                correctness1.append(c1)
                correctness3.append(c2)
                correctness5.append(c3)
                #
                # Compute matching score
                mscore = compute_matching_score(H,shape,kp1,kp2,des1,des2)
                MScore.append(mscore)


    return np.nanmean(repeatability), np.nanmean(localization_err),\
           np.nanmean(correctness1), np.nanmean(correctness3), np.nanmean(correctness5),np.nanmean(MScore)



