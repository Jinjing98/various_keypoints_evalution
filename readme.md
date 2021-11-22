about the dataset:

1.provide a directory with "ori_imgs" (whatever size bigger than 256*128 since we will only utilize the cropped one), the generate_data.py will 
1) generate 4*6 images with size 256*128 under out_imgs,
among which covering 5 transformation: blur illu scale rotation proj, each transfrom have 6 variation imges.

2) generate the cropped ori_imgs under final_ori_imgs.

3) currently we have 98 imgs under "ori_imgs" as source.

*4) we didn't consider proj transform, since I am sure how to get the correct H mat for <cropped rectangle>. It is doable if we didn't do the cropping, under this case, the transform H we did for the bigger ori img is exacty the H_GT. This doesn't holds when consider local img, but can be computed if we crop the center rectangle under rot/scale transformtion.  <This can be tested with scipt "/home/jinjing/Projects/keypoints_comparision/testH.py">
5) now we do consider the proj transformation, by getting the GT_H in another manner: modelling camera model and the imgaing processing, meaning, 2D-3D converting. reference:https://stackoverflow.com/questions/6606891/opencv-virtually-camera-rotating-translating-for-birds-eye-view


2.we consider 3 traditional methods: ORB, AGAST_SIFT, AKAZE      2 learning based methods: Superpoint KP2D

1).KP2D: require pytorch, set (confidence,top k) as you will
run the script "/home/jinjing/Projects/keypoints_comparision/brand_new/KP2D/eval_keypoint_net.py"

2).SuperPoint: require tensorflow, set (confidence, top k)
run the script "/home/jinjing/Projects/keypoints_comparision/brand_new/SuperPoint/eval_keypoint_net_SuperPoint_tf.py"

3). 3 traditional ones: due to agast_sift restriction, this requires opencv3.4.1, opencv4 may not get sift work when use agast_sift method.
run the scipt "/home/jinjing/Projects/keypoints_comparision/brand_new/SuperPoint/eval_keypoint_net_traditional.py"

4)all above three scipts have a line of code which can be uncomment to display the detected kps in img and transformed_img.

5)the metrics we consider:   Repeat, Loc error, correctness1/3/5 (depends on how we set the distance trd), Matching accu
some notes about the metrics:
	(1) when compute repeatibility: we use 3 as the distance_threshold for all the evaluation.
	(2) for learning based methods: (confidence_trd, top_k)  the top_k operation is done after filtered with confidence_trd; top_k is the max num of points get filtered.i.e the actual pts <= K.+ 
	(3) we use BF matcher,l2_distance for all the evalution; no "good_matching" parameters.




3. in order to get things work for actual temporal pair of successive frames:
1)data preparation: under directory "/home/jinjing/Projects/data_old/new_data/output/idx_video_frame/"  saved all the paired of frames we want to manul lable GT_H_mat.    The name of the pair of imgs should be: pairIdx_videoSrc_frameIdx.  pairIdx is the identity for each different pair, i.e videoSrc/frameIdx can be same under the directory.
2)annotator_main.py: this script will assist up to manually generate GT_H_Mat. Remember to keep consistent, we chose 4 pairs of points for each pair of successive img. (at least 4 pairs of pts can compute decent H_mat; if we want, we can also choose more pairs, while remember to change some code in following evaluators)
3)accrose all the different feature points methods, we have the other version of evaluator for each. Spercifically, they will be run with the "main2()"  entrance, and the called function are named with original_evalutate_func_name2().
In generall, the procedure are quite similiar as normal ones, but we only consider the detected features within polygon(formed with the chosen 4 pair of points), this function is supported by python package shapely.![trian2](https://user-images.githubusercontent.com/57319627/142914558-10e8b537-63ca-44c2-8665-7ad0eda01b35.png)
![rect1](https://user-images.githubusercontent.com/57319627/142914561-b475f2a6-0849-4a64-9af2-46eafea5a7cb.png)
 ![3_22_4 png_ORB](https://user-images.githubusercontent.com/57319627/142914887-2de28588-a121-4f0b-a1f5-b69ed88b2d07.png)
![3_22_4 png_AKAZE](https://user-images.githubusercontent.com/57319627/142914890-669721d7-bc94-4d11-a555-0b8d07ac5233.png)


