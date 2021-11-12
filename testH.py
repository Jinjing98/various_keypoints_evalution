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
path = "/home/jinjing/Projects/data/ori_imgs/5_6.png"
img = cv2.imread(path)
img = cv2.resize(img, (800,600), interpolation = cv2.INTER_AREA)
rows,cols,ch = img.shape

# #
# #
# # #
# # # cv2.warpAffine(img,M,(cols,rows))
# # M = np.array([692.6649, -254.0151, -344.57227,
# # 236.45557, 703.2224, -62219.586,
# # 0.5438765, 0.1775103, 391.67853],dtype=np.float64).reshape((3,3))
# # M = np.array([-686.3796, 341.51328, 148839.14,
# # -324.4622, -575.29474, 180949.31,
# # -0.4415784, -0.46124917, 626.00244],dtype=np.float64).reshape((3,3))
# # img2 = cv2.warpPerspective(img,M,(cols,rows))
# # cv2.imshow("",img2)
# # cv2.waitKey(0)
#
# # mat = np.load("/home/jinjing/Projects/data/out_imgs/1_1/scale/GT_H_mat.npz")["1.3"]
# # print(mat)
# # mat2 = np.load("/home/jinjing/Projects/data/out_imgs/1_1/scale_pair/SuperPoint_esti_H.npz")["1.3"]
# # print("mat: ","\n",mat2)
#
# #
# # img_mat = cv2.warpPerspective(img,mat,(cols,rows))
# # img_mat2 = cv2.warpPerspective(img,mat2,(cols,rows))
#
# # cv2.imshow("",np.concatenate((img,img_mat,img_mat2)))
# # cv2.waitKey(0)
#
#
#
#
# #
# # GT_H_mat = {}
# # for img_path in ori_img_paths:
# #     img = cv2.imread(ori_img_dir+"/"+img_path)
# #     new_img_dir = Path(new_img_dir_general+"/"+img_path[:-4], f"proj/")
# #     new_img_dir.mkdir(parents = True, exist_ok = True)
# #     new_img_dir_pair = Path(new_img_dir_general+"/"+img_path[:-4], f"proj_pair/")
# #     new_img_dir_pair.mkdir(parents = True, exist_ok = True)
#
# h,w = img.shape[:2]  # row col
# # for param in proj_list:#
# param = 5
# # H = cv2.getRotationMatrix2D((w/2,h/2),30,1)
# # H = np.vstack((H,np.array((0,0,1))))
# # get the H based on cam model
# # #remember we consider the 3*3 H for small rectangle rather for the whole frame
#
# rotXdeg = 0.1  # these two param better be smaller than 0.2, so as to get reasonable prospective
# rotYdeg = 0.1
# rotZdeg = 180   # this will control 2D rotate
# rotX = rotXdeg*np.pi/180
# rotY = rotYdeg*np.pi/180
# rotZ = rotZdeg*np.pi/180
#
# dist = 1#500  #trans_Z   # equal the [2][2] value of the final 3*3 H mat   # samller zoom inner, take the same effect as f?   #nothing 1  # better always keep fixed to 1
# f = 1.2# bigger blurer  zoom inner  # when it equals 1, eactly the same, no scaling.   # nothing 1  # potential scaling factor
# trans_X = 50
# trans_Y = 25#50
#
# #Projection 2D -> 3D matrix
# A1= np.matrix([[1, 0, -w/2],
#                [0, 1, -h/2],
#                [0, 0, 0   ],
#                [0, 0, 1   ]])
#
# # Rotation matrices around the X,Y,Z axis
# RX = np.matrix([[1,           0,            0, 0],
#                 [0,np.cos(rotX),-np.sin(rotX), 0],
#                 [0,np.sin(rotX),np.cos(rotX) , 0],
#                 [0,           0,            0, 1]])
#
# RY = np.matrix([[ np.cos(rotY), 0, np.sin(rotY), 0],
#                 [            0, 1,            0, 0],
#                 [ -np.sin(rotY), 0, np.cos(rotY), 0],
#                 [            0, 0,            0, 1]])
#
# RZ = np.matrix([[ np.cos(rotZ), -np.sin(rotZ), 0, 0],
#                 [ np.sin(rotZ), np.cos(rotZ), 0, 0],
#                 [            0,            0, 1, 0],
#                 [            0,            0, 0, 1]])
#
# #Composed rotation matrix with (RX,RY,RZ)
# R = RX * RY * RZ
#
# #Translation matrix on the Z axis change dist will change the height
# T = np.matrix([[1,0,0,trans_X],
#                [0,1,0,trans_Y],
#                [0,0,1,dist],
#                [0,0,0,1]])
#
# #Camera Intrisecs matrix 3D -> 2D
# A2= np.matrix([[f, 0, w/2,0],
#                [0, f, h/2,0],
#                [0, 0, 1,0]])
# B = T * (R * A1)
# # Final and overall transformation matrix
# H = A2 * (T * (R * A1))
#
# # w 1280 h 720
# w_r = 300
# h_r = 200
#
#
# #Projection 2D -> 3D matrix
# A1_r= np.matrix([[1, 0, -w_r/2],
#                [0, 1, -h_r/2],
#                [0, 0, 0   ],
#                [0, 0, 1   ]])
#
#
#
#
# #Camera Intrisecs matrix 3D -> 2D
# A2_r= np.matrix([[f, 0, w_r/2,0],
#                [0, f, h_r/2,0],
#                [0, 0, 1,0]])
#
#
#
#
# H_r = A2_r * (T * (R * A1_r))
#
#
#
#
#
#
#
#
#
# img_big = img.copy()
# cv2.warpPerspective(img,H,(w,h),img_big,cv2.INTER_CUBIC)
# # cv2.imshow("",np.concatenate((img,img_proj)))
# # cv2.waitKey(0)
# proj_big = img_big[int(h/2)-int(h_r/2):int(h/2)+int(h_r/2), int(w/2)-int(w_r/2):int(w/2)+int(w_r/2)]
# img_big = img[int(h/2)-int(h_r/2):int(h/2)+int(h_r/2), int(w/2)-int(w_r/2):int(w/2)+int(w_r/2)]
# proj_rect = img_big.copy()
# cv2.warpPerspective(img_big,H_r,(w_r,h_r),proj_rect,cv2.INTER_CUBIC)
# cv2.imshow("",np.concatenate((img_big,proj_big,proj_rect,proj_rect-proj_big,proj_rect-img_big)))#   first two are the 2 small ones; later two are the cropped one based on the bigger img
# cv2.waitKey(0)
# print("the difference of the H_big,H_small: ",H_r-H)
#
#




proj_list = [0.02,0.04,0.06,0.08,0.1]
for param in proj_list:#

    # there can be actual values for rotZdeg(2D rotate), f(scale),trans_X,trans_Y
    h,w = img.shape[:2]  # row col
    rotXdeg = param  # these two param better be smaller than 0.2, so as to get reasonable prospective
    rotYdeg = param
    rotZdeg = 0   # this will control 2D rotate
    rotX = rotXdeg*np.pi/180
    rotY = rotYdeg*np.pi/180
    rotZ = rotZdeg*np.pi/180
    dist = 1#  #trans_Z   # equal the [2][2] value of the final 3*3 H mat   # samller zoom inner, take the same effect as f?   #nothing 1  # better always keep fixed to 1
    f = 1#1.2# bigger blurer  zoom inner  # when it equals 1, eactly the same, no scaling.   # nothing 1  # potential scaling factor
    trans_X = 0#50
    trans_Y = 0#25#50

    #Projection 2D -> 3D matrix
    A1= np.matrix([[1, 0, -w/2],
                   [0, 1, -h/2],
                   [0, 0, 0   ],
                   [0, 0, 1   ]])

    # Rotation matrices around the X,Y,Z axis

    RX = np.matrix([[1,           0,            0, 0],
                    [0,np.cos(rotX),-np.sin(rotX), 0],
                    [0,np.sin(rotX),np.cos(rotX) , 0],
                    [0,           0,            0, 1]])

    RY = np.matrix([[ np.cos(rotY), 0, np.sin(rotY), 0],
                    [            0, 1,            0, 0],
                    [ -np.sin(rotY), 0, np.cos(rotY), 0],
                    [            0, 0,            0, 1]])

    RZ = np.matrix([[ np.cos(rotZ), -np.sin(rotZ), 0, 0],
                    [ np.sin(rotZ), np.cos(rotZ), 0, 0],
                    [            0,            0, 1, 0],
                    [            0,            0, 0, 1]])

    #Composed rotation matrix with (RX,RY,RZ)
    R = RX * RY * RZ

    #Translation matrix on the Z axis change dist will change the height
    T = np.matrix([[1,0,0,trans_X],
                   [0,1,0,trans_Y],
                   [0,0,1,dist],
                   [0,0,0,1]])

    #Camera Intrisecs matrix 3D -> 2D
    A2= np.matrix([[f, 0, w/2,0],
                   [0, f, h/2,0],
                   [0, 0, 1,0]])
    # Final and overall transformation matrix
    H_big = A2 * (T * (R * A1))

    img_proj = cv2.warpPerspective(img,H_big,(w,h))
    # cv2.imwrite(str(new_img_dir)+"/"+str(param)+".png", img_proj[int(y/2)-rec_h_half:int(y/2)+rec_h_half, int(x/2)-rec_w_half:int(x/2)+rec_w_half])#[y:y+h, x:x+w]

    h_r = 64*2#rec_h_half*2
    w_r = 128*2#rec_w_half*2
    #Projection 2D -> 3D matrix
    A1_r= np.matrix([[1, 0, -w_r/2],
                   [0, 1, -h_r/2],
                   [0, 0, 0   ],
                   [0, 0, 1   ]])

    #Camera Intrisecs matrix 3D -> 2D
    A2_r= np.matrix([[f, 0, w_r/2,0],
                   [0, f, h_r/2,0],
                   [0, 0, 1,0]])
    H_rect = A2_r * (T * (R * A1_r))
    # GT_H_mat[str(param)] = H_rect

    img_big = img.copy()
    cv2.warpPerspective(img,H_big,(w,h),img_big,cv2.INTER_CUBIC)
    # cv2.imshow("",np.concatenate((img,img_proj)))
    # cv2.waitKey(0)
    proj_big = img_big[int(h/2)-int(h_r/2):int(h/2)+int(h_r/2), int(w/2)-int(w_r/2):int(w/2)+int(w_r/2)]
    img_big = img[int(h/2)-int(h_r/2):int(h/2)+int(h_r/2), int(w/2)-int(w_r/2):int(w/2)+int(w_r/2)]
    proj_rect = img_big.copy()
    cv2.warpPerspective(img_big,H_rect,(w_r,h_r),proj_rect,cv2.INTER_CUBIC)

    # #the 5 imgs are:
    # 1) the cropped rect based on the ori big img
    # 2) the cropped rect based on the transformed big img wrt H_big
    # 3) the transformed img based on the cropped rect img,i.e img1, wrt H_rect. Notice, H_rect is the H_GT
    # 4) To prove the correctness of H_rect, we check if img2 and img3 perfect aligh by deduct img2 and img3,
    # as see in the black region, their pixel values are exact the same, this means the way we construct H_GT is correct
    # 5) for comparision, the img different of img1 and img2 shoudl be everywhere, which is the case

    cv2.imshow("",np.concatenate((img_big,proj_big,proj_rect,proj_rect-proj_big,proj_rect-img_big)))#   first two are the 2 small ones; later two are the cropped one based on the bigger img
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
