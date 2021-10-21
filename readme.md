about the dataset:

1.provide a directory with "ori_imgs" (whatever size bigger than 256*128 since we will only utilize the cropped one), the generate_data.py will 
1) generate 4*6 images with size 256*128 under out_imgs,
among which covering 4 transformation: blur illu scale rotation, each transfrom have 6 variation imges.

2) generate the cropped ori_imgs under final_ori_imgs.

3) currently we have 98 imgs under "ori_imgs" as source.

4) we didn't consider proj transform, since I am sure how to get the correct H mat for <cropped rectangle>. It is doable if we didn't do the cropping, under this case, the transform H we did for the bigger ori img is exacty the H_GT. This doesn't holds when consider local img, but can be computed if we crop the center rectangle under rot/scale transformtion.  <This can be tested with scipt "/home/jinjing/Projects/keypoints_comparision/testH.py">



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
	
