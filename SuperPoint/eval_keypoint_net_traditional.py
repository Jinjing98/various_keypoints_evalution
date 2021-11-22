import cv2
import numpy as np
from superpoint.evaluations.evaluate import evaluate_keypoint_net_ORB,evaluate_keypoint_net_AKAZE,evaluate_keypoint_net_agast_SIFT,evaluate_keypoint_net_ORB2,evaluate_keypoint_net_AKAZE2,evaluate_keypoint_net_agast_SIFT2

from termcolor import colored
# run this with opencv_env(opencv = 3.1.0)
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
    EXPER_PATH="/home/jinjing/Projects/keypoints_comparision/pretrained_models"\

    # #
    # # "orb: "
    num4features_set = [300,100]
    fast_trd_set = [10,20,30]
    for num4features in num4features_set:
        for fast_trd in fast_trd_set:
            rep, loc, c1, c3, c5, mscore = evaluate_keypoint_net_ORB(final_ori_img_dir,
                                                                     warp_img_dir,num4features,fast_trd)
            print(colored('Evaluating for ORB: num4features {} fast_trd {}'.format(num4features,fast_trd),'green'))

            print('Repeatability {0:.3f}'.format(rep))
            print('Localization Error {0:.3f}'.format(loc))
            print('Correctness d1 {:.3f}'.format(c1))
            print('Correctness d3 {:.3f}'.format(c3))
            print('Correctness d5 {:.3f}'.format(c5))
            print('MScore {:.3f}'.format(mscore))



    # # "AKAZE: "
    #     # DIFF_PM_G1       = 0,
    #     # DIFF_PM_G2       = 1,
    #     # DIFF_WEICKERT    = 2,
    #     # DIFF_CHARBONNIER = 3,
    trd_set = [5e-4,5e-3,1e-3]   # affaect the num of detected poitns
    for diff_type in range(4):
        for trd in trd_set:
            rep, loc, c1, c3, c5, mscore = evaluate_keypoint_net_AKAZE(final_ori_img_dir,warp_img_dir,
                                                                       diff_type,trd)
            print(colored('Evaluating for AKAZE: diff_type {} fast_trd {}'.format(diff_type,trd),'green'))

            print('Repeatability {0:.3f}'.format(rep))
            print('Localization Error {0:.3f}'.format(loc))
            print('Correctness d1 {:.3f}'.format(c1))
            print('Correctness d3 {:.3f}'.format(c3))
            print('Correctness d5 {:.3f}'.format(c5))
            print('MScore {:.3f}'.format(mscore))


    # rep, loc, c1, c3, c5, mscore = evaluate_keypoint_net_agast_SIFT(final_ori_img_dir,warp_img_dir)  #this func should run on opencv3.4.1!
    #"agast_SIFT"
    trds = [10,20,30]   # affaect the num of detected poitns
    agast_types = ['5_8','OAST_9_16','7_12_d','7_12_s']
    for agast_type in agast_types:
        for trd in trds:
            rep, loc, c1, c3, c5, mscore = evaluate_keypoint_net_agast_SIFT(final_ori_img_dir,warp_img_dir,
                                                                            trd,agast_type)
            print(colored('Evaluating for agast_SIFT: agast_type {} fast_trd {}'.format(agast_type,trd),'green'))

            print('Repeatability {0:.3f}'.format(rep))
            print('Localization Error {0:.3f}'.format(loc))
            print('Correctness d1 {:.3f}'.format(c1))
            print('Correctness d3 {:.3f}'.format(c3))
            print('Correctness d5 {:.3f}'.format(c5))
            print('MScore {:.3f}'.format(mscore))




def main2():


    ori_img_dir = "/home/jinjing/Projects/data/ori_imgs/"
    # ori_img_paths = os.listdir(ori_img_dir)

    warp_img_dir = "/home/jinjing/Projects/data_old/new_data/output/paired_opt/"#idx_video_frame
    final_ori_img_dir = "/home/jinjing/Projects/data_old/new_data/output/idx_video_frame/"
    EXPER_PATH="/home/jinjing/Projects/keypoints_comparision/pretrained_models"\

    # #
    # # "orb: "
    num4features_set = [300,100]
    fast_trd_set = [10,20,30]
    for num4features in num4features_set:
        for fast_trd in fast_trd_set:
            rep, loc, c1, c3, c5, mscore = evaluate_keypoint_net_ORB2(final_ori_img_dir,
                                                                     warp_img_dir,num4features,fast_trd)
            print(colored('Evaluating for ORB: num4features {} fast_trd {}'.format(num4features,fast_trd),'green'))

            print('Repeatability {0:.3f}'.format(rep))
            print('Localization Error {0:.3f}'.format(loc))
            print('Correctness d1 {:.3f}'.format(c1))
            print('Correctness d3 {:.3f}'.format(c3))
            print('Correctness d5 {:.3f}'.format(c5))
            print('MScore {:.3f}'.format(mscore))



    # # "AKAZE: "
    #     # DIFF_PM_G1       = 0,
    #     # DIFF_PM_G2       = 1,
    #     # DIFF_WEICKERT    = 2,
    #     # DIFF_CHARBONNIER = 3,
    trd_set = [5e-4,5e-3,1e-3]   # affaect the num of detected poitns
    for diff_type in range(4):
        for trd in trd_set:
            rep, loc, c1, c3, c5, mscore = evaluate_keypoint_net_AKAZE2(final_ori_img_dir,warp_img_dir,
                                                                       diff_type,trd)
            print(colored('Evaluating for AKAZE: diff_type {} fast_trd {}'.format(diff_type,trd),'green'))

            print('Repeatability {0:.3f}'.format(rep))
            print('Localization Error {0:.3f}'.format(loc))
            print('Correctness d1 {:.3f}'.format(c1))
            print('Correctness d3 {:.3f}'.format(c3))
            print('Correctness d5 {:.3f}'.format(c5))
            print('MScore {:.3f}'.format(mscore))









    # rep, loc, c1, c3, c5, mscore = evaluate_keypoint_net_agast_SIFT(final_ori_img_dir,warp_img_dir)  #this func should run on opencv3.4.1!
    #"agast_SIFT"
    trds = [10,20,30]   # affaect the num of detected poitns
    agast_types = ['5_8','OAST_9_16','7_12_d','7_12_s']
    for agast_type in agast_types:
        for trd in trds:
            rep, loc, c1, c3, c5, mscore = evaluate_keypoint_net_agast_SIFT2(final_ori_img_dir,warp_img_dir,
                                                                            trd,agast_type)
            print(colored('Evaluating for agast_SIFT: agast_type {} fast_trd {}'.format(agast_type,trd),'green'))

            print('Repeatability {0:.3f}'.format(rep))
            print('Localization Error {0:.3f}'.format(loc))
            print('Correctness d1 {:.3f}'.format(c1))
            print('Correctness d3 {:.3f}'.format(c3))
            print('Correctness d5 {:.3f}'.format(c5))
            print('MScore {:.3f}'.format(mscore))


    #







if __name__ == '__main__':
    # main()
    main2()

