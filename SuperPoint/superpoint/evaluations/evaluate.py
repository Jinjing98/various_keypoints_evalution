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

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from itertools import groupby

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


def check_each_image_has_one_correspondance(grouped_images):
    for i in grouped_images:
        assert len(i) == 2, f'Error with the image(s): {i}'
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

def evaluate_keypoint_net_SP2(ori_dir,warp_dir,sess,top_k,input_img_tensor,output_prob_nms_tensor,output_desc_tensors,conf_threshold ):
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


    images_paths = sorted(os.listdir(ori_dir))
    grouped_images = [list(i) for j, i in groupby(images_paths,
                    lambda img_name: img_name.split("_")[0])]

    check_each_image_has_one_correspondance(grouped_images)

    for i, j in grouped_images:
        img1_path = str(Path(ori_dir, i))
        img2_path = str(Path(ori_dir, j))
        npz_path = warp_dir+i[:-4]+"/GT_H_pts.npz"
        GT_H_pts = np.load(npz_path)
        GT_H = GT_H_pts["H"]
        pts1 = GT_H_pts["pts1"]
        pts2 = GT_H_pts["pts2"]

        img1 = preprocess_image(img1_path)
        # pridict!
        out1 = sess.run([output_prob_nms_tensor, output_desc_tensors],
                        feed_dict={input_img_tensor: np.expand_dims(img1, 0)})



        keypoint_map1 = np.squeeze(out1[0])
        descriptor_map1 = np.squeeze(out1[1])

        kp1, des1 = extract_superpoint_keypoints_and_descriptors(
                keypoint_map1, descriptor_map1, top_k,confidence=conf_threshold)



        img2 =  preprocess_image(img2_path)
        shape = img2.shape
        distance_thresh = 3
        out2 = sess.run([output_prob_nms_tensor, output_desc_tensors],
                        feed_dict={input_img_tensor: np.expand_dims(img2, 0)})
        keypoint_map2 = np.squeeze(out2[0])
        descriptor_map2 = np.squeeze(out2[1])
# des2 300*256
        kp2, des2 = extract_superpoint_keypoints_and_descriptors(
                keypoint_map2, descriptor_map2, top_k,confidence=conf_threshold) # for leared feature, the conf and top k are already filtered here

        #filtered kps accroding to the polygon:
        w1,h1,w2,h2,w3,h3,w4,h4 = list(pts1.reshape((8,)))
        w1_2,h1_2,w2_2,h2_2,w3_2,h3_2,w4_2,h4_2 = list(pts2.reshape((8,)))
        # print("k:",list(pts1.reshape((6,))))
        polygon1 = Polygon([(w1, h1), (w2, h2), (w3, h3),(w4,h4)])
        polygon2 = Polygon([(w1_2, h1_2), (w2_2, h2_2), (w3_2, h3_2),(w4_2,h4_2)])

        # ori_img0 = cv2.imread(img1_path)
        # outimg0 = ori_img0.copy()
        # cv2.drawKeypoints(ori_img0,kp1,outimg0)
        # # cv2.imshow("",outimg)
        # # cv2.waitKey(0)


        new_kp1 = []
        new_kp2 = []
        new_des1 = []
        new_des2 = []
        for i,pt in enumerate(kp1):
            point1 =  Point(pt.pt[0],pt.pt[1])
            if polygon1.contains(point1)==True:
                new_kp1.append(pt)
                new_des1.append(des1[i])
                # print("add pt1",point1)
        new_des1 = np.array(new_des1).reshape(-1,256)
        for i,pt in enumerate(kp2):
            point2 =  Point(pt.pt[0],pt.pt[1])
            if polygon2.contains(point2)==True:
                new_kp2.append(pt)
                new_des2.append(des2[i])
                # print("add pt1",point1)
        new_des2 = np.array(new_des2).reshape(-1,256)


        ori_img = cv2.imread(img1_path)
        outimg = ori_img.copy()
        cv2.drawKeypoints(ori_img,new_kp1,outimg)
        # cv2.imshow("",outimg)
        # cv2.waitKey(0)

        ori_img2 = cv2.imread(img2_path)
        outimg2 = ori_img2.copy()
        cv2.drawKeypoints(ori_img2,new_kp2,outimg2)



        #  read the pts1 pts2 from the files then vis pts within the trian
        outimg = cv2.circle(outimg, (w1,h1), radius=0, color=(0, 0, 255), thickness=5)
        outimg = cv2.circle(outimg, (w2,h2), radius=0, color=(0, 0, 255), thickness=5)
        outimg = cv2.circle(outimg, (w3,h3), radius=0, color=(0, 0, 255), thickness=5)
        outimg = cv2.circle(outimg, (w4,h4), radius=0, color=(0, 0, 255), thickness=5)
        outimg2 = cv2.circle(outimg2, (w1_2,h1_2), radius=0, color=(0, 0, 255), thickness=5)
        outimg2 = cv2.circle(outimg2, (w2_2,h2_2), radius=0, color=(0, 0, 255), thickness=5)
        outimg2 = cv2.circle(outimg2, (w3_2,h3_2), radius=0, color=(0, 0, 255), thickness=5)
        outimg2 = cv2.circle(outimg2, (w4_2,h4_2), radius=0, color=(0, 0, 255), thickness=5)
        # cv2.imshow("",np.concatenate((img.get(),warpped_img.get())))



        #
        # cv2.imshow("",np.concatenate((outimg,outimg2),axis = 1))
        # cv2.waitKey(0)
        # #kp1 kp2 should be fixed

        N1, N2, rep, loc_err= compute_repeatability(new_kp1,new_kp2,GT_H,shape,distance_thresh = 3)



        repeatability.append(rep)
        localization_err.append(loc_err)

        # Compute correctness
        # c1, c2, c3 = compute_homography(data, keep_k_points=top_k)
        c1, c2, c3 = compute_homography(GT_H,shape,new_kp1,new_kp2,new_des1,new_des2)
        correctness1.append(c1)
        correctness3.append(c2)
        correctness5.append(c3)

        # Compute matching score
        mscore = compute_matching_score(GT_H,shape,new_kp1,new_kp2,new_des1,new_des2)
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




def evaluate_keypoint_net_ORB2(ori_dir,warp_dir,num4features,fast_trd):
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

    orb = cv2.ORB_create(nfeatures=num4features,fastThreshold=fast_trd)




    images_paths = sorted(os.listdir(ori_dir))
    grouped_images = [list(i) for j, i in groupby(images_paths,
                    lambda img_name: img_name.split("_")[0])]

    check_each_image_has_one_correspondance(grouped_images)

    for i_name, j_name in grouped_images:
        img1_path = str(Path(ori_dir, i_name))
        img2_path = str(Path(ori_dir, j_name))
        npz_path = warp_dir+i_name[:-4]+"/GT_H_pts.npz"
        GT_H_pts = np.load(npz_path)
        GT_H = GT_H_pts["H"]
        pts1 = GT_H_pts["pts1"]
        pts2 = GT_H_pts["pts2"]

        img1 = cv2.imread(img1_path)


        kp1, des1 = orb.detectAndCompute(img1,None)  # maybe less than 500



        img2 =  cv2.imread(img2_path)
        shape = img2.shape
        distance_thresh = 3

# des2 300*256
        kp2, des2 = orb.detectAndCompute(img2,None)

        #filtered kps accroding to the polygon:
        w1,h1,w2,h2,w3,h3,w4,h4 = list(pts1.reshape((8,)))
        w1_2,h1_2,w2_2,h2_2,w3_2,h3_2,w4_2,h4_2 = list(pts2.reshape((8,)))
        # print("k:",list(pts1.reshape((6,))))
        polygon1 = Polygon([(w1, h1), (w2, h2), (w3, h3),(w4,h4)])
        polygon2 = Polygon([(w1_2, h1_2), (w2_2, h2_2), (w3_2, h3_2),(w4_2,h4_2)])

        # ori_img0 = cv2.imread(img1_path)
        # outimg0 = ori_img0.copy()
        # cv2.drawKeypoints(ori_img0,kp1,outimg0)
        # # cv2.imshow("",outimg)
        # # cv2.waitKey(0)


        new_kp1 = []
        new_kp2 = []
        new_des1 = []
        new_des2 = []
        for i,pt in enumerate(kp1):
            point1 =  Point(pt.pt[0],pt.pt[1])
            if polygon1.contains(point1)==True:
                new_kp1.append(pt)
                new_des1.append(des1[i])
                # print("add pt1",point1)
        new_des1 = np.array(new_des1).reshape(-1,32)
        for i,pt in enumerate(kp2):
            point2 =  Point(pt.pt[0],pt.pt[1])
            if polygon2.contains(point2)==True:
                new_kp2.append(pt)
                new_des2.append(des2[i])
                # print("add pt1",point1)
        new_des2 = np.array(new_des2).reshape(-1,32)


        ori_img = cv2.imread(img1_path)
        outimg = ori_img.copy()
        cv2.drawKeypoints(ori_img,new_kp1,outimg)
        # cv2.imshow("",outimg)
        # cv2.waitKey(0)

        ori_img2 = cv2.imread(img2_path)
        outimg2 = ori_img2.copy()
        cv2.drawKeypoints(ori_img2,new_kp2,outimg2)



        #  read the pts1 pts2 from the files then vis pts within the trian
        outimg = cv2.circle(outimg, (w1,h1), radius=0, color=(0, 0, 255), thickness=5)
        outimg = cv2.circle(outimg, (w2,h2), radius=0, color=(0, 0, 255), thickness=5)
        outimg = cv2.circle(outimg, (w3,h3), radius=0, color=(0, 0, 255), thickness=5)
        outimg = cv2.circle(outimg, (w4,h4), radius=0, color=(0, 0, 255), thickness=5)
        outimg2 = cv2.circle(outimg2, (w1_2,h1_2), radius=0, color=(0, 0, 255), thickness=5)
        outimg2 = cv2.circle(outimg2, (w2_2,h2_2), radius=0, color=(0, 0, 255), thickness=5)
        outimg2 = cv2.circle(outimg2, (w3_2,h3_2), radius=0, color=(0, 0, 255), thickness=5)
        outimg2 = cv2.circle(outimg2, (w4_2,h4_2), radius=0, color=(0, 0, 255), thickness=5)

        a = np.concatenate((outimg,outimg2),axis = 1)
        # cv2.imwrite("/home/jinjing/Projects/data_old/new_data/output/temp/"+str(i_name)+"_ORB.png",a)







        # cv2.imshow("",np.concatenate((outimg,outimg2),axis = 1))
        # cv2.waitKey(0)
        # #kp1 kp2 should be fixed



        N1, N2, rep, loc_err= compute_repeatability(new_kp1,new_kp2,GT_H,shape,distance_thresh = 3)



        repeatability.append(rep)
        localization_err.append(loc_err)

        # Compute correctness
        # c1, c2, c3 = compute_homography(data, keep_k_points=top_k)
        c1, c2, c3 = compute_homography(GT_H,shape,new_kp1,new_kp2,new_des1,new_des2)
        correctness1.append(c1)
        correctness3.append(c2)
        correctness5.append(c3)

        # Compute matching score
        mscore = compute_matching_score(GT_H,shape,new_kp1,new_kp2,new_des1,new_des2)
        MScore.append(mscore)

    return np.nanmean(repeatability), np.nanmean(localization_err),\
           np.nanmean(correctness1), np.nanmean(correctness3), np.nanmean(correctness5),np.nanmean(MScore)




def evaluate_keypoint_net_AKAZE2(ori_dir,warp_dir,diff_type,trd):
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


    akaze = cv2.AKAZE_create(threshold=trd,diffusivity=diff_type)
    # akaze = cv2.AKAZE_create()



    images_paths = sorted(os.listdir(ori_dir))
    grouped_images = [list(i) for j, i in groupby(images_paths,
                    lambda img_name: img_name.split("_")[0])]

    check_each_image_has_one_correspondance(grouped_images)

    for i_name, j_name in grouped_images:
        img1_path = str(Path(ori_dir, i_name))
        img2_path = str(Path(ori_dir, j_name))
        npz_path = warp_dir+i_name[:-4]+"/GT_H_pts.npz"
        GT_H_pts = np.load(npz_path)
        GT_H = GT_H_pts["H"]
        pts1 = GT_H_pts["pts1"]
        pts2 = GT_H_pts["pts2"]

        img1 = cv2.imread(img1_path)


        kp1, des1 = akaze.detectAndCompute(img1,None)  # maybe less than 500



        img2 =  cv2.imread(img2_path)
        shape = img2.shape
        distance_thresh = 3

# des2 300*256
        kp2, des2 = akaze.detectAndCompute(img2,None)

        #filtered kps accroding to the polygon:
        w1,h1,w2,h2,w3,h3,w4,h4 = list(pts1.reshape((8,)))
        w1_2,h1_2,w2_2,h2_2,w3_2,h3_2,w4_2,h4_2 = list(pts2.reshape((8,)))
        # print("k:",list(pts1.reshape((6,))))
        polygon1 = Polygon([(w1, h1), (w2, h2), (w3, h3),(w4,h4)])
        polygon2 = Polygon([(w1_2, h1_2), (w2_2, h2_2), (w3_2, h3_2),(w4_2,h4_2)])



        new_kp1 = []
        new_kp2 = []
        new_des1 = []
        new_des2 = []
        for i,pt in enumerate(kp1):
            point1 =  Point(pt.pt[0],pt.pt[1])
            if polygon1.contains(point1)==True:
                new_kp1.append(pt)
                new_des1.append(des1[i])
                # print("add pt1",point1)
        new_des1 = np.array(new_des1).reshape(-1,61)
        for i,pt in enumerate(kp2):
            point2 =  Point(pt.pt[0],pt.pt[1])
            if polygon2.contains(point2)==True:
                new_kp2.append(pt)
                new_des2.append(des2[i])
                # print("add pt1",point1)
        new_des2 = np.array(new_des2).reshape(-1,61)


        ori_img = cv2.imread(img1_path)
        outimg = ori_img.copy()
        cv2.drawKeypoints(ori_img,new_kp1,outimg)
        # cv2.imshow("",outimg)
        # cv2.waitKey(0)

        ori_img2 = cv2.imread(img2_path)
        outimg2 = ori_img2.copy()
        cv2.drawKeypoints(ori_img2,new_kp2,outimg2)



        #  read the pts1 pts2 from the files then vis pts within the trian
        outimg = cv2.circle(outimg, (w1,h1), radius=0, color=(0, 0, 255), thickness=5)
        outimg = cv2.circle(outimg, (w2,h2), radius=0, color=(0, 0, 255), thickness=5)
        outimg = cv2.circle(outimg, (w3,h3), radius=0, color=(0, 0, 255), thickness=5)
        outimg = cv2.circle(outimg, (w4,h4), radius=0, color=(0, 0, 255), thickness=5)
        outimg2 = cv2.circle(outimg2, (w1_2,h1_2), radius=0, color=(0, 0, 255), thickness=5)
        outimg2 = cv2.circle(outimg2, (w2_2,h2_2), radius=0, color=(0, 0, 255), thickness=5)
        outimg2 = cv2.circle(outimg2, (w3_2,h3_2), radius=0, color=(0, 0, 255), thickness=5)
        outimg2 = cv2.circle(outimg2, (w4_2,h4_2), radius=0, color=(0, 0, 255), thickness=5)


        a = np.concatenate((outimg,outimg2),axis = 1)
        # cv2.imwrite("/home/jinjing/Projects/data_old/new_data/output/temp/"+str(i_name)+"_AKAZE.png",a)
        # cv2.imshow("",np.concatenate((outimg,outimg2),axis = 1))
        # cv2.waitKey(0)
        # # #kp1 kp2 should be fixed



        N1, N2, rep, loc_err= compute_repeatability(new_kp1,new_kp2,GT_H,shape,distance_thresh = 3)



        repeatability.append(rep)
        localization_err.append(loc_err)

        # Compute correctness
        # c1, c2, c3 = compute_homography(data, keep_k_points=top_k)
        c1, c2, c3 = compute_homography(GT_H,shape,new_kp1,new_kp2,new_des1,new_des2)
        correctness1.append(c1)
        correctness3.append(c2)
        correctness5.append(c3)

        # Compute matching score
        mscore = compute_matching_score(GT_H,shape,new_kp1,new_kp2,new_des1,new_des2)
        MScore.append(mscore)


    return np.nanmean(repeatability), np.nanmean(localization_err),\
           np.nanmean(correctness1), np.nanmean(correctness3), np.nanmean(correctness5),np.nanmean(MScore)



def evaluate_keypoint_net_agast_SIFT2(ori_dir,warp_dir,trd,agast_type):
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




    images_paths = sorted(os.listdir(ori_dir))
    grouped_images = [list(i) for j, i in groupby(images_paths,
                    lambda img_name: img_name.split("_")[0])]

    check_each_image_has_one_correspondance(grouped_images)

    for i_name, j_name in grouped_images:
        img1_path = str(Path(ori_dir, i_name))
        img2_path = str(Path(ori_dir, j_name))
        npz_path = warp_dir+i_name[:-4]+"/GT_H_pts.npz"
        GT_H_pts = np.load(npz_path)
        GT_H = GT_H_pts["H"]
        pts1 = GT_H_pts["pts1"]
        pts2 = GT_H_pts["pts2"]

        img1 = cv2.imread(img1_path)

        kp1 = agast.detect(img1)  #  how to restrict the num of detected kps
        kp1,des1 = sift.compute(img1,kp1)




        img2 =  cv2.imread(img2_path)
        shape = img2.shape
        distance_thresh = 3

        kp2 = agast.detect(img2)  #  how to restrict the num of detected kps
        kp2,des2 = sift.compute(img2,kp2)

        #filtered kps accroding to the polygon:
        w1,h1,w2,h2,w3,h3,w4,h4 = list(pts1.reshape((8,)))
        w1_2,h1_2,w2_2,h2_2,w3_2,h3_2,w4_2,h4_2 = list(pts2.reshape((8,)))
        # print("k:",list(pts1.reshape((6,))))
        polygon1 = Polygon([(w1, h1), (w2, h2), (w3, h3),(w4,h4)])
        polygon2 = Polygon([(w1_2, h1_2), (w2_2, h2_2), (w3_2, h3_2),(w4_2,h4_2)])



        new_kp1 = []
        new_kp2 = []
        new_des1 = []
        new_des2 = []
        for i,pt in enumerate(kp1):
            point1 =  Point(pt.pt[0],pt.pt[1])
            if polygon1.contains(point1)==True:
                new_kp1.append(pt)
                new_des1.append(des1[i])
                # print("add pt1",point1)
        new_des1 = np.array(new_des1).reshape(-1,128)
        for i,pt in enumerate(kp2):
            point2 =  Point(pt.pt[0],pt.pt[1])
            if polygon2.contains(point2)==True:
                new_kp2.append(pt)
                new_des2.append(des2[i])
                # print("add pt1",point1)
        new_des2 = np.array(new_des2).reshape(-1,128)


        ori_img = cv2.imread(img1_path)
        outimg = ori_img.copy()
        cv2.drawKeypoints(ori_img,new_kp1,outimg)
        # cv2.imshow("",outimg)
        # cv2.waitKey(0)

        ori_img2 = cv2.imread(img2_path)
        outimg2 = ori_img2.copy()
        cv2.drawKeypoints(ori_img2,new_kp2,outimg2)



        #  read the pts1 pts2 from the files then vis pts within the trian
        outimg = cv2.circle(outimg, (w1,h1), radius=0, color=(0, 0, 255), thickness=5)
        outimg = cv2.circle(outimg, (w2,h2), radius=0, color=(0, 0, 255), thickness=5)
        outimg = cv2.circle(outimg, (w3,h3), radius=0, color=(0, 0, 255), thickness=5)
        outimg = cv2.circle(outimg, (w4,h4), radius=0, color=(0, 0, 255), thickness=5)
        outimg2 = cv2.circle(outimg2, (w1_2,h1_2), radius=0, color=(0, 0, 255), thickness=5)
        outimg2 = cv2.circle(outimg2, (w2_2,h2_2), radius=0, color=(0, 0, 255), thickness=5)
        outimg2 = cv2.circle(outimg2, (w3_2,h3_2), radius=0, color=(0, 0, 255), thickness=5)
        outimg2 = cv2.circle(outimg2, (w4_2,h4_2), radius=0, color=(0, 0, 255), thickness=5)
        a = np.concatenate((outimg,outimg2),axis = 1)
        # cv2.imwrite("/home/jinjing/Projects/data_old/new_data/output/temp/"+str(i_name)+"_AGAST_SIFT.png",a)

        # #kp1 kp2 should be fixed



        N1, N2, rep, loc_err= compute_repeatability(new_kp1,new_kp2,GT_H,shape,distance_thresh = 3)



        repeatability.append(rep)
        localization_err.append(loc_err)

        # Compute correctness
        # c1, c2, c3 = compute_homography(data, keep_k_points=top_k)
        c1, c2, c3 = compute_homography(GT_H,shape,new_kp1,new_kp2,new_des1,new_des2)
        correctness1.append(c1)
        correctness3.append(c2)
        correctness5.append(c3)

        # Compute matching score
        mscore = compute_matching_score(GT_H,shape,new_kp1,new_kp2,new_des1,new_des2)
        MScore.append(mscore)




    return np.nanmean(repeatability), np.nanmean(localization_err),\
           np.nanmean(correctness1), np.nanmean(correctness3), np.nanmean(correctness5),np.nanmean(MScore)


