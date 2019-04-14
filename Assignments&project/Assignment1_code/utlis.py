import numpy as np
import cv2
import copy

def splitRGB(img):

    R = copy.deepcopy(img)
    G = copy.deepcopy(img)
    B = copy.deepcopy(img)

    for i in range(img.shape[0]):
        for j in  range(img.shape[1]):
            if ((i % 2 == 0 and j % 2 == 0) or (i % 2 != 0 and j % 2 != 0 )):
                R[(i, j)]= 0
            if(i % 2 !=0 or j % 2 != 0):
                B[(i, j)]= 0
            if (i % 2 == 0 or j % 2 == 0):
                G[(i, j)] = 0
    return R,G,B

def AppliedFilter(R,G,B):

    filter1=np.array([[0.25, 0.5, 0.25],
                      [0.5, 1, 0.5],
                      [0.25, 0.5, 0.25]])

    filter4=np.array([[ 0, 0.25, 0],
                      [0.25, 1,  0.25],
                      [0,   0.25, 0]])

    R_filtered = cv2.filter2D(R,-1,filter4)
    G_flitered= cv2.filter2D(G,-1, filter1)
    B_filtered = cv2.filter2D(B, -1, filter1)

    return R_filtered,G_flitered,B_filtered

def median(img1,img2):
    #
    # A = img1-img2
    # A_m = cv2.medianBlur(A, 3)
    #
    # Dst=A_m+img2

    A=cv2.subtract(img1,img2)
    B=cv2.subtract(img2,img1)

    A_m=cv2.medianBlur(A,3)
    B_m=cv2.medianBlur(B,3)

    Dst = cv2.add(A_m,img2)
    Dst = cv2.subtract(Dst,B_m)

    return Dst

def CmpDiffrence(img_orignal,img_filted):

    r_cha = cv2.subtract(img_orignal[:, :, 0], img_filted[:, :, 0])
    g_cha = cv2.subtract(img_orignal[:, :, 1], img_filted[:, :, 1])
    b_cha = cv2.subtract(img_orignal[:, :, 2], img_filted[:, :, 2])

    diff = r_cha**2+g_cha**2+b_cha**2

    return diff
