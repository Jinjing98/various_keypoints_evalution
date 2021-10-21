import cv2
import time
import pathlib
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

import cv2
import numpy as np


# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# from match_features_demo import  extract_superpoint_keypoints_and_descriptors, preprocess_image
# #comment below line of code, the code will run in GPU mode
# # # this line of code will let the code to run on CPU, but this have to be in the very beginning of the code
# # tf.config.set_visible_devices([], 'GPU')
# #
# #
# #below line of code will display the log for debug to see if it is real GPU
# # To find out which devices your operations and tensors are assigned to
# # tf.debugging.set_log_device_placement(True)


# # below code may be needed if we want the code to run on GPU correctly.
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             #below is an important line of code for enabling gpu
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs are: ",gpus, "  ",len(logical_gpus), "Logical GPUs are :",logical_gpus)
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)
















def create_rot_new(ori_img_paths,ori_img_dir,new_img_dir_general ):
    GT_H_mat = {}
    for img_path in ori_img_paths:
        img = cv2.imread(ori_img_dir+"/"+img_path)
        # new_img_dir = new_img_dir+"/"+img_path[:-4]+"/rot/"
        new_img_dir = Path(new_img_dir_general+"/"+img_path[:-4], f"rot/")
        new_img_dir.mkdir(parents = True, exist_ok = True)
        new_img_dir_pair = Path(new_img_dir_general+"/"+img_path[:-4], f"rot_pair/")
        # new_img_dir_pair.mkdir(parents = True, exist_ok = True)

        y,x = img.shape[:2]
        for ang in rot_list:
            rotate = cv2.getRotationMatrix2D((x/2,y/2),ang,1)  # the 3rd row is  0 0 1
            small_rotate = cv2.getRotationMatrix2D((rec_w_half,rec_h_half),ang,1)
            #  it is the same as get affine transform?
            img_rot = cv2.warpAffine(img,rotate,(x,y))
            # cv2.imshow("",img_rot)
            # cv2.waitKey(0)
            cv2.imwrite(str(new_img_dir)+"/"+str(ang)+".png", img_rot[int(y/2)-rec_h_half:int(y/2)+rec_h_half, int(x/2)-rec_w_half:int(x/2)+rec_w_half])#[y:y+h, x:x+w]
            #rotate is wrong for cropped img!
            GT_H_mat[str(ang)] = np.vstack((small_rotate,np.array((0,0,1))))#jj
        GT_H_mat_path = Path(new_img_dir,f"GT_H_mat.npz")
        np.savez(GT_H_mat_path,**GT_H_mat)

def create_scale_new(ori_img_paths,ori_img_dir,new_img_dir_general ):
    GT_H_mat = {}
    for img_path in ori_img_paths:
        img = cv2.imread(ori_img_dir+"/"+img_path)
        # new_img_dir = new_img_dir+"/"+img_path[:-4]+"/scale/"
        new_img_dir = Path(new_img_dir_general+"/"+img_path[:-4], f"scale/")
        new_img_dir.mkdir(parents = True, exist_ok = True)
        new_img_dir_pair = Path(new_img_dir_general+"/"+img_path[:-4], f"scale_pair/")
        # new_img_dir_pair.mkdir(parents = True, exist_ok = True)

        y,x = img.shape[:2]
        for scale in scale_list:
            rotate = cv2.getRotationMatrix2D((x/2,y/2),0,scale)  # the 3rd row is  0 0 1
            small_rotate = cv2.getRotationMatrix2D((rec_w_half,rec_h_half),0,scale)
            img_scale = cv2.warpAffine(img,rotate,(x,y))
            cv2.imwrite(str(new_img_dir)+"/"+str(scale)+".png", img_scale[int(y/2)-rec_h_half:int(y/2)+rec_h_half, int(x/2)-rec_w_half:int(x/2)+rec_w_half])#[y:y+h, x:x+w]
            GT_H_mat[str(scale)] = np.vstack((small_rotate,np.array((0,0,1))))
        GT_H_mat_path = Path(new_img_dir,f"GT_H_mat.npz")
        np.savez(GT_H_mat_path,**GT_H_mat)


def create_illu_new(ori_img_paths,ori_img_dir,new_img_dir_general ):# gamma
    GT_H_mat = {}
    for img_path in ori_img_paths:
        img = cv2.imread(ori_img_dir+"/"+img_path)
        # new_img_dir = new_img_dir+"/"+img_path[:-4]+"/illu/"
        new_img_dir = Path(new_img_dir_general+"/"+img_path[:-4], f"illu/")
        new_img_dir.mkdir(parents = True, exist_ok = True)
        new_img_dir_pair = Path(new_img_dir_general+"/"+img_path[:-4], f"illu_pair/")
        # new_img_dir_pair.mkdir(parents = True, exist_ok = True)

        y,x = img.shape[:2]

        for gamma in illu_list:
            invGamma = 1.0 /gamma
            table = np.array([((i / 255.0) ** invGamma) * 255
                                for i in np.arange(0, 256)]).astype("uint8")
            img_illu = cv2.LUT(img, table)
            cv2.imwrite(str(new_img_dir)+"/"+str(gamma)+".png", img_illu[int(y/2)-rec_h_half:int(y/2)+rec_h_half, int(x/2)-rec_w_half:int(x/2)+rec_w_half])#[y:y+h, x:x+w]
            rotate = cv2.getRotationMatrix2D((x/2,y/2),0,1)
            GT_H_mat[str(gamma)] = np.vstack((rotate,np.array((0,0,1))))
        GT_H_mat_path = Path(new_img_dir,f"GT_H_mat.npz")
        np.savez(GT_H_mat_path,**GT_H_mat)


def create_blur_new(ori_img_paths,ori_img_dir,new_img_dir_general ):
    GT_H_mat = {}
    for img_path in ori_img_paths:
        img = cv2.imread(ori_img_dir+"/"+img_path)
        # new_img_dir = new_img_dir+"/"+img_path[:-4]+"/blur/"
        new_img_dir = Path(new_img_dir_general+"/"+img_path[:-4], f"blur/")
        new_img_dir.mkdir(parents = True, exist_ok = True)
        new_img_dir_pair = Path(new_img_dir_general+"/"+img_path[:-4], f"blur_pair/")
        # new_img_dir_pair.mkdir(parents = True, exist_ok = True)

        y,x = img.shape[:2]
        for size in blur_list:#[(2,2),(4,4),(6,6),(8,8),(10,10),(12,12)]:
            size = (size,size)
            img_blur = cv2.blur(img,size)
            cv2.imwrite(str(new_img_dir)+"/"+str(size[0])+".png", img_blur[int(y/2)-rec_h_half:int(y/2)+rec_h_half, int(x/2)-rec_w_half:int(x/2)+rec_w_half])#[y:y+h, x:x+w]
            rotate = cv2.getRotationMatrix2D((x/2,y/2),0,1)
            GT_H_mat[str(size[0])] = np.vstack((rotate,np.array((0,0,1))))
        GT_H_mat_path = Path(new_img_dir,f"GT_H_mat.npz")
        np.savez(GT_H_mat_path,**GT_H_mat)

#proj mix have problem?
def create_proj_new(ori_img_paths,ori_img_dir,new_img_dir_general ):
    GT_H_mat = {}
    for img_path in ori_img_paths:
        img = cv2.imread(ori_img_dir+"/"+img_path)
        new_img_dir = Path(new_img_dir_general+"/"+img_path[:-4], f"proj/")
        new_img_dir.mkdir(parents = True, exist_ok = True)
        new_img_dir_pair = Path(new_img_dir_general+"/"+img_path[:-4], f"proj_pair/")
        # new_img_dir_pair.mkdir(parents = True, exist_ok = True)

        y,x = img.shape[:2]
        for param in proj_list:#
            H = cv2.getRotationMatrix2D((x/2,y/2),0,1)
            H = np.vstack((H,np.array(( 0.0001*param,0.0001*param,1))))
            # small_rotate = cv2.getRotationMatrix2D((rec_w_half,rec_h_half),0,1)
            GT_H_mat[str(param)] = H
            img_proj = cv2.warpPerspective(img,H,(x,y))
            cv2.imwrite(str(new_img_dir)+"/"+str(param)+".png", img_proj[int(y/2)-rec_h_half:int(y/2)+rec_h_half, int(x/2)-rec_w_half:int(x/2)+rec_w_half])#[y:y+h, x:x+w]

        GT_H_mat_path = Path(new_img_dir,f"GT_H_mat.npz")
        np.savez(GT_H_mat_path,**GT_H_mat)




def create_mix_new(ori_img_paths,ori_img_dir,new_img_dir_general):
    GT_H_mat = {}
    for img_path in ori_img_paths:
        img = cv2.imread(ori_img_dir+"/"+img_path)
        # new_img_dir = new_img_dir+"/"+img_path[:-4]+"/blur/"
        new_img_dir = Path(new_img_dir_general+"/"+img_path[:-4], f"mix/")
        new_img_dir.mkdir(parents = True, exist_ok = True)
        new_img_dir_pair = Path(new_img_dir_general+"/"+img_path[:-4], f"mix_pair/")
        # new_img_dir_pair.mkdir(parents = True, exist_ok = True)

        y,x = img.shape[:2]
        H_unit = cv2.getRotationMatrix2D((x/2,y/2),0,1)

        for p_rot,p_scale,p_illu,p_blur,p_proj in zip(rot_list,scale_list,illu_list,blur_list,proj_list):
            rotate_scale = cv2.getRotationMatrix2D((x/2,y/2),p_rot,p_scale)  # the 3rd row is  0 0 1
            H_rotate_scale = np.vstack((rotate_scale,np.array(( 0,0,1))))
            H_proj = np.vstack((H_unit,np.array(( 0.0001*p_proj,0.0001*p_proj,1))))
            H_rotate_scale_proj = np.matmul(H_proj,H_rotate_scale)

            size = (p_blur,p_blur)
            # invGamma = 1.0 /p_illu
            # table = np.array([((i / 255.0) ** invGamma) * 255
            #                     for i in np.arange(0, 256)]).astype("uint8")

            # blur_img = cv2.blur(img,size)
            # blur_illu_img = cv2.LUT(blur_img, table)
            mix_img = cv2.warpPerspective(img,H_rotate_scale_proj,(x,y))
            # str(p_rot)+"_"+str(p_scale)+"_"+str(p_blur)+"_"+str(p_illu)+"_"+str(p_proj)
            cv2.imwrite(str(new_img_dir)+"/"+str(p_rot)+"_"+str(p_scale)+"_"+str(p_blur)+"_"+str(p_illu)+"_"+str(p_proj) +".png", mix_img[int(y/2)-rec_h_half:int(y/2)+rec_h_half, int(x/2)-rec_w_half:int(x/2)+rec_w_half])#[y:y+h, x:x+w]
            GT_H_mat[str(p_rot)+"_"+str(p_scale)+"_"+str(p_blur)+"_"+str(p_illu)+"_"+str(p_proj)] = H_rotate_scale_proj

        GT_H_mat_path = Path(new_img_dir,f"GT_H_mat.npz")
        np.savez(GT_H_mat_path,**GT_H_mat)


def create_final_ori(ori_img_paths,ori_img_dir,final_ori_img_dir):


    new_img_dir = Path(final_ori_img_dir)
    new_img_dir.mkdir(parents = True, exist_ok = True)
    for img_path in ori_img_paths:
        img = cv2.imread(ori_img_dir+"/"+img_path)
        y,x = img.shape[:2]
        cv2.imwrite(str(final_ori_img_dir)+"/"+img_path, img[int(y/2)-rec_h_half:int(y/2)+rec_h_half, int(x/2)-rec_w_half:int(x/2)+rec_w_half])



def get_kp_des_match(transform,transform_params,kp1_des1_np,kp1,des1,method):
        kps_des_mat = {}
        matches_mat = {}
        H_mat = {}
        kps_des_mat[str(0)] = kp1_des1_np
        for new_img_param in transform_params:
            new_img_name = str(new_img_dir)+"/"+img_path[:-4]+"/"+transform+"/"+str(new_img_param)+".png"
            new_img_color = cv2.imread(str(new_img_name))
            new_img = cv2.cvtColor(new_img_color,cv2.COLOR_BGR2GRAY)
            if method == "ORB":
                kp2, des2 = orb.detectAndCompute(new_img,None)
            if method == "GFTT_SIFT":
                kp2 = cv2.goodFeaturesToTrack(new_img, maxCorners=num_feature, qualityLevel=0.01,minDistance=10)#max_corners=25, quality_level=0.01, min_distance=10, detection_size=1
                kp2 = [cv2.KeyPoint(k[0][0], k[0][1], 1) for k in kp2]  # 1 is detection size ?
                kp2, des2 = sift.compute(new_img, kp2)   #  200*128
            if method == "AGAST_SIFT":
                kp2 = agast.detect(new_img)  #  how to restrict the num of detected kps
                kp2,des2 = sift.compute(new_img,kp2)
            if method == "SuperPoint":
                new_img_cal = preprocess_image(str(new_img_name))
                    # pridict!
                out2 = sess.run([output_prob_nms_tensor, output_desc_tensors],
                                    feed_dict={input_img_tensor: np.expand_dims(new_img_cal, 0)})
                keypoint_map2 = np.squeeze(out2[0])
                descriptor_map2 = np.squeeze(out2[1])

                kp2, des2 = extract_superpoint_keypoints_and_descriptors(
                        keypoint_map2, descriptor_map2, keep_k_best)






            if len(kp2) > dict_trd_num_kps2:   #  not put in dict if kps2 num too less
                kp2_des2_np = np.zeros((len(kp2),2+des2.shape[1]))
                id = 0
                for kp in kp2:
                    kp2_des2_np[id] = [kp.pt[0],kp.pt[1]]+list(des2[id])
                    id += 1
                kps_des_mat[str(new_img_param)] = kp2_des2_np

                if method == "ORB":
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                if method == "GFTT_SIFT" or method == "AGAST_SIFT" or method == "SuperPoint":
                    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)





                matches = bf.match(des1,des2)
                matches = sorted(matches, key = lambda x:x.distance)
                num4matches = len(matches)
                matches = matches[:num4matches]#?
                # print(matches[0].queryIdx)

                idx = 0
                matches_np = np.empty((num4matches,7))
                for match in matches:
                    # matches_np[idx] = [match.queryIdx,match.trainIdx,match.distance]
                    #kp1[m.queryIdx].pt

                    matches_np[idx] = [match.queryIdx,kp1[match.queryIdx].pt[0],kp1[match.queryIdx].pt[1],match.trainIdx,kp2[match.trainIdx].pt[0],kp2[match.trainIdx].pt[1],match.distance]

                    idx += 1
                matches_mat[str(new_img_param)] = matches_np

                # if len(matches) <8:
                #     print("Warning! "+str(len(matches))+" is less than 8 matches!"+"\n the path is : "+ new_img_name)

                img3 = cv2.drawMatches(img,kp1,new_img,kp2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                img3 = cv2.resize(img3,(img.shape[1],int(img.shape[0]/2)))#720 1280   1280 320
                matched_img_path = str(new_img_dir)+"/"+img_path[:-4]+"/"+transform+"_pair/"+str(new_img_param)+"_"+method+".png"
                cv2.imwrite(matched_img_path,img3)

                src_pts = matches_np[:,1:3]
                dst_pts = matches_np[:,4:6]

                H,status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                if H is not None:
                    H_mat[str(new_img_param)] = H

            else:
                print("less than ",dict_trd_num_kps2," keypoints detected in this img2! ignore this frames pair! ","img: ",img_path[:-4]," setting: ",method,transform,new_img_param,)
            #     kps_des_mat[str(new_img_param)] = None
            #     matches_mat[str(new_img_param)] = None
            #     H_mat[str(new_img_param)] = None





        des_mat_path = str(new_img_dir)+"/"+img_path[:-4]+"/"+transform+"_pair/"+method+"_kps_des"+".npz"
        matches_mat_path = str(new_img_dir)+"/"+img_path[:-4]+"/"+transform+"_pair/"+method+"_matches"+".npz"
        np.savez(des_mat_path,**kps_des_mat)
        np.savez(matches_mat_path,**matches_mat)
        H_mat_path = str(new_img_dir)+"/"+img_path[:-4]+"/"+transform+"_pair/"+method+"_esti_H"+".npz"
        np.savez(H_mat_path,**H_mat)




if __name__=="__main__":

 

    
    ori_img_dir = "/home/jinjing/Projects/data/ori_imgs"

    new_img_dir = "/home/jinjing/Projects/data/out_imgs"
    final_ori_img_dir = "/home/jinjing/Projects/data/final_ori_imgs"
    EXPER_PATH="/home/jinjing/Projects/keypoints_comparision/pretrained_models"


    
    
    
    
    rec_h_half = 64  #unit of 8.  e.g 64
    rec_h_half = 64  #unit of 8.  e.g 64
    rec_w_half = 128#  e.g. 128
    dict_trd_num_kps2 = 5
    rot_list = [10,15,20,80,85,90]
    scale_list = [0.7,0.8,0.9,1.1,1.2,1.3]  # zoom out
    blur_list = [2,3,4,5,6,7]
    illu_list = [0.4,0.6,0.8,1.2,1.4,1.6]
    proj_list = [1,2,3,-1,-2,-3]
    mix_list = [str(p_rot)+"_"+str(p_scale)+"_"+str(p_blur)+"_"+str(p_illu)+"_"+str(p_proj) for p_rot,p_scale,p_blur,p_illu,p_proj in zip(rot_list,scale_list,blur_list,illu_list,proj_list)]#str(p_rot)+"_"+str(p_scale)+"_"+str(p_illu)+"_"+str(p_blur)
    transform_params_list = [rot_list,scale_list,blur_list,illu_list,proj_list,mix_list]
    method_list = ["SuperPoint","ORB","AGAST_SIFT","GFTT_SIFT"]
    transform_list = ["rot","scale","blur","illu","proj","mix"]


    ori_img_paths = os.listdir(ori_img_dir)
    create_rot_new(ori_img_paths,ori_img_dir,new_img_dir)
    create_scale_new(ori_img_paths,ori_img_dir,new_img_dir)
    create_illu_new(ori_img_paths,ori_img_dir,new_img_dir)
    create_blur_new(ori_img_paths,ori_img_dir,new_img_dir)
    # create_proj_new(ori_img_paths,ori_img_dir,new_img_dir)
    # create_mix_new(ori_img_paths,ori_img_dir,new_img_dir)

    create_final_ori(ori_img_paths,ori_img_dir,final_ori_img_dir)

    # a short cut to reduce the number of editing
    ori_img_dir = final_ori_img_dir


    print("data collection done.\n start processing ...")






    # num_feature = 200

    # #load model
    # parser = argparse.ArgumentParser()
    # parser = argparse.ArgumentParser(description='Compute the homography \
    #         between two images with the SuperPoint feature matches.')
    # parser.add_argument('--k_best', type=int, default=200,
    #                     help='Maximum number of keypoints to keep \
    #                     (default: 1000)')
    # args = parser.parse_args()

    # weights_name = "sp_v6"#args.weights_name""
    # keep_k_best = args.k_best

    # weights_root_dir = Path(EXPER_PATH, 'saved_models')
    # weights_root_dir.mkdir(parents=True, exist_ok=True)
    # weights_dir = Path(weights_root_dir, weights_name)

    # graph = tf.Graph()
    # with tf.Session(graph=graph) as sess:
    #     tf.saved_model.loader.load(sess,
    #                                [tf.saved_model.tag_constants.SERVING],#r"E:\Google Drive\\files.sem3\NCT\Reuben_lab\keypoint_detector_descriptor_evaluator-main\models\SuperPoint\pretrained_models\sp_v6.tar")
    #                                str(weights_dir))

        # input_img_tensor = graph.get_tensor_by_name('superpoint/image:0')
        # output_prob_nms_tensor = graph.get_tensor_by_name(
        #     'superpoint/prob_nms:0')
        # output_desc_tensors = graph.get_tensor_by_name(
        #     'superpoint/descriptors:0')


        # for method in method_list:

            # for img_path in ori_img_paths:
            #     # all4img = list(Path(new_img_dir+img_path[:-4],f"rot\\").rglob("*.png"))
            #     img = cv2.imread(ori_img_dir+"/"+img_path)
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #     # img = preprocess_image(ori_img_dir+"/"+img_path)

                # if method == "ORB":
                #     orb = cv2.ORB_create(nfeatures=num_feature)
                #     kp1, des1 = orb.detectAndCompute(img,None)  # maybe less than 500
                # if method == "GFTT_SIFT":
                #     sift = cv2.xfeatures2d.SIFT_create()
                #     kp1 = cv2.goodFeaturesToTrack(img, maxCorners=num_feature, qualityLevel=0.01,minDistance=10)#max_corners=25, quality_level=0.01, min_distance=10, detection_size=1
                #     kp1 = [cv2.KeyPoint(k[0][0], k[0][1], 1) for k in kp1]  # 1 is detection size
                #     kp1,des1 = sift.compute(img, kp1)   #  200*128

                # if method == "AGAST_SIFT":  # how to constraint the num of feature here?
                #     sift = cv2.xfeatures2d.SIFT_create()
                #     AGAST_TYPES = {
                #                     '5_8': cv2.AgastFeatureDetector_AGAST_5_8,
                #                     'OAST_9_16': cv2.AgastFeatureDetector_OAST_9_16,
                #                     '7_12_d': cv2.AgastFeatureDetector_AGAST_7_12d,
                #                     '7_12_s': cv2.AgastFeatureDetector_AGAST_7_12s }

                #     agast = cv2.AgastFeatureDetector_create(threshold=10, nonmaxSuppression=True, type=AGAST_TYPES['OAST_9_16'])
                #     kp1 = agast.detect(img)
                #     kp1,des1 = sift.compute(img,kp1)
                # if method == "SuperPoint":
                #         img1 = preprocess_image(ori_img_dir+"/"+img_path)
                #         # pridict!
                #         out1 = sess.run([output_prob_nms_tensor, output_desc_tensors],
                #                         feed_dict={input_img_tensor: np.expand_dims(img1, 0)})
                #         keypoint_map1 = np.squeeze(out1[0])
                #         descriptor_map1 = np.squeeze(out1[1])

                        # kp1, des1 = extract_superpoint_keypoints_and_descriptors(
                        #     keypoint_map1, descriptor_map1, keep_k_best)

                # num4kps = len(kp1)
                # kp1_des1_np = np.empty((num4kps,2+des1.shape[1]))
                # id = 0
                # for kp in kp1:
                #     kp1_des1_np[id] = [kp.pt[0],kp.pt[1]]+list(des1[id])
                #     id += 1
                # kps_des_mat = {}
                # matches_mat = {}
                # kps_des_mat[str(0)] = kp1_des1_np
                # for transform,transform_params in zip(transform_list,transform_params_list):
                #     print(transform,transform_params)
                #     get_kp_des_match(transform,transform_params,kp1_des1_np,kp1,des1,method)














