import cv2
import numpy as np

import math
import sys
import random
from scipy import ndimage
from KeyPoints_detector import *
from MOPS import *
MIN_MATCH_COUNT = 10
M = np.eye(3)

def project(f1,H,matches):
    # prject 返回array形式的数据
    #[len(matches),2]

    x2 = np.zeros((len(matches),2))
    for i in range(len(matches)):
        ma = matches[i]
        (x1,y1) = f1[ma.queryIdx].pt

        p = np.array([x1,y1,1])
        ptrans = np.dot(H,p)
        x2[i][0] = int(ptrans[0]/ptrans[2])
        x2[i][1] = int(ptrans[1]/ptrans[2])

    return x2
def countInlier_numbers(H, matches, f1, f2, inlierThreshold):
    inlier_inx = []
    total_numInlier = 0

    for i in range(len(matches)):

        ma = matches[i]
        (x1, y1) = f1[ma.queryIdx].pt
        (x2, y2) = f2[ma.trainIdx].pt

        # Project Function
        p = np.array([x1, y1, 1])
        ptrans = np.dot(H, p)
        ptrans[0] = ptrans[0] / ptrans[2]
        ptrans[1] = ptrans[1] / ptrans[2]

        x_diff = (x2 - ptrans[0]) * (x2 - ptrans[0])
        y_diff = (y2 - ptrans[1]) * (y2 - ptrans[1])
        distance = np.sqrt(x_diff + y_diff)
        if distance <= inlierThreshold:
            total_numInlier +=1
            inlier_inx.append(matches[i])
    return total_numInlier,inlier_inx
def RANSAC(matches,RandomSize,iteration_num, RANSACthresh,f1, f2):
    RandomSize=4
    maxinliers = 0
    result_linersmatch = []

    for i in range(iteration_num):
        randmatch = np.random.choice(matches,RandomSize)
        #random select 4 matched pair points

        H = np.eye(3)
    #compute H
        H = findHomography(f1, f2, randmatch)
    #获得inliermatch
        inliersmatch = Inliers_matches_index(f1, f2, matches, H, RANSACthresh)
    #保留最大inlier
        if len(inliersmatch)>maxinliers:
            maxinliers = len(inliersmatch)
            result_linersmatch = inliersmatch
    #重新计算次 全局H
    H = Recompute_H(f1, f2, matches, result_linersmatch)

    return H
class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position #H
def findHomography(f1, f2, matches, A_out=None):

    num_matches = len(matches)

    num_rows = 2 * num_matches
    num_cols = 9
    A_matrix_shape = (num_rows,num_cols)
    A = np.zeros(A_matrix_shape)

    for i in range(len(matches)):
        m = matches[i]
        (a_x, a_y) = f1[m.queryIdx].pt
        (b_x, b_y) = f2[m.trainIdx].pt

        A[2*i] = [a_x , a_y, 1, 0, 0, 0, -b_x * a_x, -b_x * a_y, -b_x]
        A[2*i+1] = [0, 0, 0, a_x, a_y, 1, -b_y * a_x, -b_y * a_y, -b_y]

    #Strange matrix
    U, s, Vt = np.linalg.svd(A)
    if A_out is not None:
        A_out[:] = A

    #Homography to be calculated
    H = np.eye(3)
    #Fill the homography H with the appropriate elements of the SVD

    lastIdx = Vt.shape[0]
    H = (Vt[lastIdx-1] / Vt[lastIdx-1][8] ).reshape(3,3)

    return H
def Inliers_matches_index(f1, f2, matches, M, RANSACthresh):

    inlier_indices = []

    for i in range(len(matches)):

        ma = matches[i]
        (x1,y1) = f1[ma.queryIdx].pt
        (x2,y2) = f2[ma.trainIdx].pt

        #transform into homogeneous coordinates
        p = np.array([x1,y1,1])
        ptrans = np.dot(M,p)
        ptrans[0] = ptrans[0]/ptrans[2]
        ptrans[1] = ptrans[1]/ptrans[2]

        x_diff = (x2 - ptrans[0])*(x2 - ptrans[0])
        y_diff = (y2 - ptrans[1])*(y2 - ptrans[1])
        distance = np.sqrt(x_diff + y_diff)
        if distance <= RANSACthresh:
            inlier_indices.append(i)
    return inlier_indices
def Recompute_H(f1, f2, matches, inlier_indices):
    #Stitch_function homography matrix M
    M = np.eye(3)

    newmatches = []
    for i in range(len(inlier_indices)):
        newmatches.append(matches[inlier_indices[i]])

    # def change_f2kp(f1,f2,matches):
    #     for item in matches:
    # M=cv2.findHomography(kp1,kp2)
    M = findHomography(f1, f2, newmatches)

    return M
def canvas4points(image,homInv):
    #This function is for images to inverse back to stitchedImage
    # '1.计算stitchedImage的size。(两张尺寸/多张尺寸)把Img2的四个角投影到Img1上,
    # 用来确定投影后的stitchedImage的大小,用到project和InvM
    #图像位置


    #四个点的坐标 转换成HOMO形式 [x,y,1]
    height, width = image.shape[:2]
    O = np.array([0, 0, 1])
    A = np.array([0, height - 1, 1])
    B = np.array([width - 1, height - 1, 1])
    C = np.array([width - 1, 0, 1])

    O = np.dot(homInv, O)
    A = np.dot(homInv, A)
    B = np.dot(homInv, B)
    C = np.dot(homInv, C)

    def up(x):
        if x < 0.0:
            x = math.floor(x)
            #Return the floor of x as a float,
            #the largest integer value less than or equal to x.
        else:
            x = math.ceil(x)
        return x

    minX = up(min(O[0] / O[2], A[0] / A[2], B[0] / B[2], C[0] / C[2]))
    minY = up(min(O[1] / O[2], A[1] / A[2], B[1] / B[2], C[1] / C[2]))
    maxX = up(max(O[0] / O[2], A[0] / A[2], B[0] / B[2], C[0] / C[2]))
    maxY = up(max(O[1] / O[2], A[1] / A[2], B[1] / B[2], C[1] / C[2]))

    return minX,minY,maxX,maxY
def acc_Blend(img, acc, M, blendWidth):
    #img*M

    h = img.shape[0]
    w = img.shape[1]

    h_acc = acc.shape[0]
    w_acc = acc.shape[1]

    ## get the boundary of img2 of canvas
    minX, minY, maxX, maxY = canvas4points(img, M)

    for i in range(minX, maxX, 1):
        for j in range(minY, maxY, 1):

            p = np.array([i, j, 1.])
            p = np.dot(np.linalg.inv(M), p)
            projected_x = min(p[0] / p[2], w - 1)
            projected_y = min(p[1] / p[2], h - 1)
            # 超出边界
            if projected_x < 0 or projected_x >= w or projected_y < 0 or projected_y >= h:
                continue
            #黑
            if ((img[int(projected_y), int(projected_x), 0] == 0) and (img[int(projected_y), int(projected_x), 1] == 0) and \
                (img[int(projected_y), int(projected_x), 2] == 0)):
                continue
            # Blend pos/blendwith
            if projected_x >= 0 and projected_x < w - 1 and projected_y >= 0 and projected_y < h - 1:
                weight = 1.0
                if projected_x >= minX and projected_x < minX + blendWidth:
                    weight = 1. * (projected_x - minX) / blendWidth
                if projected_x <= maxX and projected_x > maxX - blendWidth:
                    weight = 1. * (maxX - projected_x) / blendWidth
                acc[j, i, 3] += weight
                #取三通道 0,1,2 乘以weight
                for k in range(3):
                    # pixel =cv2.getRectSubPix(img,(1,1),img[i][j])
                    # print("pixel",pixel)
                    acc[j, i, k] += img[int(projected_y), int(projected_x), k] * weight
                    # acc[j, i, k] += pixel * weight
def help_function_blend(img_lst, translation, width):
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(img_lst):
        if count != 0 and count != (len(img_lst) - 1):
            continue

        M = i.position

        M_trans = translation.dot(M)

        p = np.array([0.5 * width, 0, 1])
        p = M_trans.dot(p)

        # First image
        if count == 0:
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(img_lst) - 1):
            x_final, y_final = p[:2] / p[2]

    return x_init, y_init, x_final, y_final
def adjust_Blend(acc):

    h_acc = acc.shape[0]
    w_acc = acc.shape[1]
    img = np.zeros((h_acc, w_acc, 3))

    for i in range(0, w_acc, 1):
        for j in range(0, h_acc, 1):
            if acc[j,i,3]>0:
                img[j,i,0] = int (acc[j,i,0] / acc[j,i,3])
                img[j,i,1] = int (acc[j,i,1] / acc[j,i,3])
                img[j,i,2] = int (acc[j,i,2] / acc[j,i,3])
            else:
                img[j,i,0] = 0
                img[j,i,1] = 0
                img[j,i,2] = 0
    img = np.uint8(img)
    return img
def acc_img_for_blend(img_lst, translation, blendWidth, canvasWidth, canvasaHeight, channels):
    acc = np.zeros((canvasaHeight, canvasWidth, channels + 1))
    # Add in all the images
    H = np.identity(3)
    for count,i in enumerate(img_lst):
        H = i.position #H and img
        img = i.img

        M_trans = translation.dot(H)
        acc_Blend(img, acc, M_trans, blendWidth)

    return acc
def compute_canvas_Size(img_lst):

    # Compute bounding box for the whole pics
    minX = sys.maxsize
    minY = sys.maxsize
    maxX = 0
    maxY = 0
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)
    for i in img_lst:
        M = i.position
        img = i.img
        h, w,c= img.shape

        if channels == -1:
            channels = c
            width = w
        minx,miny,maxx,maxy = canvas4points(img,M) #ookk
        minX = min(minX,minx)
        minY = min(minY,miny)
        maxX = max(maxX,maxx)
        maxY = max(maxY,maxy)
    # Create an accumulator image
    canvasWeight = int(math.ceil(maxX) - math.floor(minX))
    canvasHeight = int(math.ceil(maxY) - math.floor(minY))
    print('Canvas Width and Height:', (canvasWeight, canvasHeight))
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

    return canvasWeight, canvasHeight, channels, width,translation
def blending(img_lst, blendWidth, A_out=None):

    accWidth, accHeight, channels, width, translation = compute_canvas_Size(img_lst)

    acc = acc_img_for_blend(img_lst, translation, blendWidth, accWidth, accHeight, channels)

    compImage = adjust_Blend(acc)

    # Determine the final image width
    outputWidth =  accWidth
    x_init, y_init, x_final, y_final = help_function_blend(img_lst, translation, width)
    # Compute the affine transform
    A = np.identity(3)

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR)

    return croppedImage
def ORB(leftImage,rightImage):
    # Try opencv ORB
    leftGrey = cv2.cvtColor(leftImage, cv2.COLOR_BGR2GRAY)
    rightGrey = cv2.cvtColor(rightImage, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    leftKeypoints, leftDescriptors = orb.detectAndCompute(leftGrey, None)
    rightKeypoints, rightDescriptors = orb.detectAndCompute(rightGrey, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(leftDescriptors, rightDescriptors)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:300]

    return matches,leftKeypoints,rightKeypoints
def compute_H(leftImage, rightImage, flag=None):

#1.Harris corner + SIFT

    if (flag == 'ORB'):
        #try other feature detector function.Not use.
        matches, leftKeypoints, rightKeypoints = ORB(leftImage, rightImage)
    if(flag == 'MOPS'):
        img1_gray = cv2.cvtColor(leftImage, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(rightImage, cv2.COLOR_BGR2GRAY)

        kp1,kp2 = Harris_corner(leftImage,rightImage)
        des1 =MPOSdescribtor(leftImage,kp1)
        des2 =MPOSdescribtor(rightImage,kp2)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)
        print("match pairs:", len(matches))
        # 存的是 一对k=2

        # store all the good_match matches as per Lowe's ratio test.
        good_match = []
        # parameter =np.array([0.2,0.2,0.5,0.2,0.1]) #0.2 ok
        i = 0
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good_match.append(m)
        i += 1

        if len(good_match) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_match]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_match]).reshape(-1, 1, 2)

        else:
            print("couldn't find enough matches points - %d/%d" % (len(good_match), MIN_MATCH_COUNT))
            matchesMask = None

        matches =good_match
        leftKeypoints=kp1
        rightKeypoints=kp2

    else:
        matches, leftKeypoints, rightKeypoints = sift(leftImage, rightImage)

    img3 = cv2.drawMatches(leftImage, leftKeypoints, rightImage, rightKeypoints, matches, None)
    cv2.imwrite("matches_for_twopairs.png",img3)

#2.RANSANC
    iteration =1000 #iteration
    RANSACThreshold =5.0 #threhold
    M = RANSAC(matches, 4, iteration, RANSACThreshold, leftKeypoints, rightKeypoints)
    img3 = cv2.drawMatches(leftImage, leftKeypoints, rightImage, rightKeypoints, matches, None)
    cv2.imwrite("after_RANSAC.png",img3)
    return M
def warpPerspectiveFunction(img,M,weight,height):
    dst = np.zeros((weight,height,3))
    for i in range(weight):
        for j in  range(height):
            p1 = np.array([i,j,1])
            ptrans = np.dot(M, p1)
            ptrans[0] = ptrans[0] / ptrans[2]
            ptrans[1] = ptrans[1] / ptrans[2]
            newX = int(ptrans[0])
            newY = int(ptrans[1])
            if newX < 0 or newX >= img.shape[0]  or newY < 0 or newY >= img.shape[1] :
                continue
            else: dst[i,j,:] = img[newX,newY,:]
    return dst
def Stitch_function(leftImage, rightImage, Flag=None):
    left = leftImage
    right = rightImage
    #comput_H
    H = compute_H(left, right, Flag)
    Hinv = np.linalg.inv(H)

    H /= Hinv[2, 2] #f9

#求变换后的picture框

    minX, minY, maxX, maxY=canvas4points(rightImage, Hinv)
    # print("inX, minY, maxX, maxY",minX, minY, maxX, maxY)
    # Create an accumulator image

    newWidth = int(np.ceil(maxX) - np.floor(minX))
    newHeight = int(np.ceil(maxY) - np.floor(minY))

    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

    # warpedRightImage1 = cv2.warpPerspective(
    #     right, np.dot(translation, H), (newWidth, newHeight)
    # )
    warpedRightImage = warpPerspectiveFunction(right, np.dot(translation, H),newWidth, newHeight)

    # warpedLeftImage1 = cv2.warpPerspective(
    #     left, translation, (newWidth, newHeight)
    # )
    warpedLeftImage = warpPerspectiveFunction(left,translation, newWidth, newHeight)

    alpha = 0.5
    beta = 1.0 - alpha
    #cv2.addWeighted: dst = src1*alpha + src2*beta + gamma;

    dst = cv2.addWeighted( warpedLeftImage, alpha, warpedRightImage, beta, 0.0)
    cv2.imwrite("dst.png",dst)
    #blend
    image_lst = []
    H_initial = np.eye(3)
    image_lst.append(ImageInfo('', leftImage, np.linalg.inv(H_initial)))
    image_lst.append(ImageInfo('', rightImage, Hinv))
    return blending(image_lst, 20)



