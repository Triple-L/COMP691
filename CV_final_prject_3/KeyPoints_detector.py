import cv2
import numpy as np

import math
import sys
import random
from scipy import ndimage
from scipy.ndimage import filters

def computeHarrisValues(Image):

    height, width = Image.shape[:2]

    harrisImage = np.zeros(Image.shape[:2], dtype=float)
    orientationImage = np.zeros(Image.shape[:2], dtype=float)

    sobx = np.zeros(Image.shape[:2], dtype=float)
    filters.sobel(Image, 1, sobx)
    soby = np.zeros(Image.shape[:2], dtype=float)
    filters.sobel(Image, 0, soby)
    # sobx = filters.convolve(srcImage,sx,mode='reflect')
    # soby = filters.convolve(srcImage,sy,mode='reflect')
    Ix = sobx*sobx
    Iy = soby*soby
    Ixy = sobx*soby


    Wxx = filters.gaussian_filter(Ix,sigma=0.5)
    Wyy = filters.gaussian_filter(Iy,sigma=0.5)
    Wxy = filters.gaussian_filter(Ixy,sigma=0.5)

    harrisImage = Wxx*Wyy - Wxy*Wxy - 0.1*(Wxx+Wyy)*(Wxx+Wyy)
    orientationImage  = np.arctan2(soby,sobx)*(180) / np.pi

    return harrisImage, orientationImage
def computeLocalMaxima(harrisImage):
    '''
    Input:
        harrisImage -- numpy array containing the Harris score at
                       each pixel.
    Output:
        destImage -- numpy array containing True/False at
                     each pixel, depending on whether
                     the pixel value is the local maxima in
                     its 7x7 neighborhood.
                     :type harrisImage: object
    '''
    height, width = harrisImage.shape[:2]
    destImage = np.zeros_like(harrisImage, np.bool)

    newmax = ndimage.maximum_filter(harrisImage,size=7)
    for i in range(height):
        for j in range(width):
            # newmax[i,j] = np.max(newpd[i:i+7,j:j+7])
            if harrisImage[i,j]==newmax[i,j]:
                destImage[i,j] = True
            else:
                destImage[i,j] = False
    return destImage
def detectKeypoints(image):

    image = image.astype(np.float32)
    image /= 255.
    height, width = image.shape[:2]
    features = []

    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    harrisImage, orientationImage = computeHarrisValues(grayImage)
    #harrisImage=[harrisImage>np.average(harrisImage)]
    harrisMaxImage =computeLocalMaxima(harrisImage)


    threhold = 0.01*np.max(harrisImage)

    #print(0.01*np.max(harrisImage),np.average(harrisImage))

    for y in range(height):
        for x in range(width):
            if not harrisMaxImage[y, x]:
                continue

            f = cv2.KeyPoint()
            f.size = 10
            f.angle = orientationImage[y,x]
            f.pt = (x,y)
            f.response = harrisImage[y,x]
            if(f.response>threhold):
                features.append(f)
    return features
def Harris_corner(img1,img2):
    features1 = detectKeypoints(img1)
    features2 = detectKeypoints(img2)
    # print("features1 len:",str(len(features1)))
    # print("features2 len:", str(len(features2)))
    return features1,features2
def sift(img1,img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # sift.Stitch_function()
    kp1,kp2 = Harris_corner(img1,img2)
    # des1=MPOSdescribtor(img1, kp1)
    # des2=MPOSdescribtor(img2, kp2)

    # print("des1:",des1)
    # print("des2:",des2)
    kp11,des1 = sift.compute(img1_gray, kp1)
    kp22,des2 = sift.compute(img2_gray, kp2)

    '''
    # find the keypoints and descriptors with SIFT
    # kp1 kp2 keypoints des1 128个 方向特征
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    '''


    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)
    print("match pairs:",len(matches))
    # 存的是 一对k=2

    # store all the good_match matches as per Lowe's ratio test.
    good_match = []
    # parameter =np.array([0.2,0.2,0.5,0.2,0.1]) #0.2 ok
    i = 0
    for m, n in matches:
        if m.distance < 0.8*n.distance:
            good_match.append(m)
    i+=1

    if len(good_match) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_match]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_match]).reshape(-1, 1, 2)

    else:
        print("couldn't find enough matches points - %d/%d" % (len(good_match), MIN_MATCH_COUNT))
        matchesMask = None


    return good_match,kp1,kp2