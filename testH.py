import cv2
import numpy as np
from math import cos,sin
#
# path = "E:\Datasets\surgical\ori_imgs\\4_5.png"
# path = "/home/jinjing/Projects/data/new_data/output/v_haha_jj_oo/1.png"
# img = cv2.imread(path)
# rows,cols,ch = img.shape
# # pts1 = np.float32([[50,50],[200,50],[50,200],[50,20]])
# # # pts2 = np.float32([[10,100],[200,50],[100,250],[50,80]])
# # # M1 = cv2.getAffineTransform(pts1,pts2)  # 3 pairs
# # # M2 = cv2.findHomography(pts1,pts2)  # 6 pairs
# # print(M2)
# # M3 = cv2.getPerspectiveTransform(pts1,pts2)  # 4 pairs
# # print(M3)
# # M1 = np.array([1,0,0,
# #                0,1,0,
# #                -0.0003,-0.000,1],dtype=np.float64).reshape((3,3))
# # # print(M1)
# # dst = cv2.warpPerspective(img,M1,(cols,rows))  # Only receive 2*3 affine transformation
# # # #
# # dst = np.concatenate((img, dst), axis=1)
# cv2.imshow("",img)
# cv2.waitKey(0)
#
#
# # kp2_new = np.matmul(M1,np.hstack((3,4,1)))
# # print("kkkk",kp2_new)
# # kp2_new = (kp2_new/kp2_new[2])[:2]
# # print("lkl;k;l",kp2_new)
#
#
# #
# # cv2.warpAffine(img,M,(cols,rows))
# M = np.array([692.6649, -254.0151, -344.57227,
# 236.45557, 703.2224, -62219.586,
# 0.5438765, 0.1775103, 391.67853],dtype=np.float64).reshape((3,3))
# M = np.array([-686.3796, 341.51328, 148839.14,
# -324.4622, -575.29474, 180949.31,
# -0.4415784, -0.46124917, 626.00244],dtype=np.float64).reshape((3,3))
# img2 = cv2.warpPerspective(img,M,(cols,rows))
# cv2.imshow("",img2)
# cv2.waitKey(0)



# test Jinjing   80
path = "/home/jinjing/Projects/data/final_ori_imgs/1_1.png"
img = cv2.imread(path)
rows,cols,ch = img.shape

#
#
# #
# # cv2.warpAffine(img,M,(cols,rows))
# M = np.array([692.6649, -254.0151, -344.57227,
# 236.45557, 703.2224, -62219.586,
# 0.5438765, 0.1775103, 391.67853],dtype=np.float64).reshape((3,3))
# M = np.array([-686.3796, 341.51328, 148839.14,
# -324.4622, -575.29474, 180949.31,
# -0.4415784, -0.46124917, 626.00244],dtype=np.float64).reshape((3,3))
# img2 = cv2.warpPerspective(img,M,(cols,rows))
# cv2.imshow("",img2)
# cv2.waitKey(0)

mat = np.load("/home/jinjing/Projects/data/out_imgs/1_1/scale/GT_H_mat.npz")["1.3"]
print(mat)
mat2 = np.load("/home/jinjing/Projects/data/out_imgs/1_1/scale_pair/SuperPoint_esti_H.npz")["1.3"]
print("mat: ","\n",mat2)


img_mat = cv2.warpPerspective(img,mat,(cols,rows))
img_mat2 = cv2.warpPerspective(img,mat2,(cols,rows))

cv2.imshow("",np.concatenate((img,img_mat,img_mat2)))
cv2.waitKey(0)













#
# y,x = img.shape[:2]   #128 256
# rotate = cv2.getRotationMatrix2D((x/2,y/2),0,2)  #  20~19*(x/2) 19*(y/2)
# print(x,y,"\n",rotate)
# img2 = cv2.warpAffine(img,rotate,(x,y))
#
# # get the center of demo img
#
#
# x2 = x/2 - (256/4)/2
# y2 = y/2 - (128/4)/2
# crop_img = img[int(y2):int(y2+(128/4)), int(x2):int(x2+(256/4))]
# cv2.imshow("",img)
# cv2.waitKey(0)
# cv2.imshow("",crop_img)
# cv2.waitKey(0)
#
#
# # # cv2.imshow("",img2)
# # # cv2.waitKey(0)
# # cv2.imshow("",np.concatenate((img,img2)))
# # cv2.waitKey(0)
