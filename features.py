import math

import cv2
import numpy as np
import scipy
from scipy import ndimage
from scipy.ndimage import filters

from scipy.spatial import distance

def inbounds(shape, indices):
    assert len(shape) == len(indices)
    for i, ind in enumerate(indices):
        if ind < 0 or ind >= shape[i]:
            return False
    return True


## Keypoint detectors ######

def saveHarrisImage(harrisImage, srcImage):
    '''
    Input:
        srcImage -- Grayscale image in a numpy array with
                    values in [0, 1].
        harrisImage -- Grayscale input image in a numpy array with
                    values in [0, 1].
    '''
    outshape = [harrisImage.shape[0], harrisImage.shape[1], 3]
    outImage = np.zeros(outshape)
    srcNorm = srcImage * (0.3 * 255 / (np.max(srcImage) + 1e-50))
    outImage[:, :, :] = np.expand_dims(srcNorm, 2)
    # Add in the harris keypoints as red
    outImage[:, :, 2] += harrisImage * (4 * 255 / (np.max(harrisImage)) + 1e-50)
    cv2.imwrite("harris.png", outImage)

    # Compute harris values of an image.
def computeHarrisValues(srcImage):
    '''
    Input:
        srcImage -- Grayscale input image in a numpy array with
                    values in [0, 1]. The dimensions are (rows, cols).
    Output:
        harrisImage -- numpy array containing the Harris score at
                       each pixel.
        orientationImage -- numpy array containing the orientation of the
                            gradient at each pixel in degrees.
    '''
    height, width = srcImage.shape[:2]

    harrisImage = np.zeros(srcImage.shape[:2],dtype=float)
    orientationImage = np.zeros(srcImage.shape[:2],dtype=float)

    # 1: Compute the harris corner strength for 'srcImage' at
    # each pixel and store in 'harrisImage'.  See the project page
    # for direction on how to do this. Also compute an orientation
    # for each pixel and store it in 'orientationImage.'


    # sx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    # sy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

    sobx = np.zeros(srcImage.shape[:2],dtype=float)
    filters.sobel(srcImage,1,sobx)
    soby = np.zeros(srcImage.shape[:2],dtype=float)
    filters.sobel(srcImage,0,soby)
    # sobx = filters.convolve(srcImage,sx,mode='reflect')
    # soby = filters.convolve(srcImage,sy,mode='reflect')
    Ix = sobx*sobx
    Iy = soby*soby
    Ixy = sobx*soby


    Wxx = filters.gaussian_filter(Ix,sigma=0.5)
    Wyy = filters.gaussian_filter(Iy,sigma=0.5)
    Wxy = filters.gaussian_filter(Ixy,sigma=0.5)


    # for i in range(height):
    #     for j in range(width):
    #         M = np.array([[Wxx[i,j],Wxy[i,j]],[Wxy[i,j],Wyy[i,j]]])
    #         R = np.linalg.det((M)-0.1*np.trace(M)*np.trace(M))
    #         harrisImage[i,j] = R
    #         orientationImage[i, j] = np.arctan2(Ix[i, j], Iy[i, j]) * (180) / np.pi
            # orientationImage[i,j] = np.arctan2(Ix[i,j],Iy[i,j])
    harrisImage = Wxx*Wyy - Wxy*Wxy - 0.1*(Wxx+Wyy)*(Wxx+Wyy)
    orientationImage  = np.arctan2(soby,sobx)*(180) / np.pi

    saveHarrisImage(harrisImage, srcImage)

    return harrisImage, orientationImage

def checkBorder( va, borderA, vb, borderB):
        if va - 1 >= 0 and va + 1 < borderA and vb - 1 >= 0 and vb + 1 < borderB:
            return True
        else:
            return False



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

    # newpd = np.zeros((height+6,width+6),dtype=float)
    # newpd[3:3+height,3:3+width] = harrisImage
    # newmax = np.zeros(height,width)

    # 2: Compute the local maxima image
    newmax = ndimage.maximum_filter(harrisImage,size=7)
    for i in range(height):
        for j in range(width):
            # newmax[i,j] = np.max(newpd[i:i+7,j:j+7])
            if harrisImage[i,j]==newmax[i,j]:
                destImage[i,j] = True
            else:
                destImage[i,j] = False
    # for y in range(height):
    #     for x in range(width):
    #         destImage[y,x] = True
    #         for j in range(-3,4):
    #             for i in range(-3,4):
    #                 if 0<=y+i<height and 0<=x+j<width and harrisImage[y+i,x+j]>harrisImage[y, x]:
    #                     destImage[y, x] = False
    return destImage

def detectKeypoints(image):
    '''
    Input:
        image -- BGR image with values between [0, 255]
    Output:
        list of detected keypoints, fill the cv2.KeyPoint objects with the
        coordinates of the detected keypoints, the angle of the gradient
        (in degrees), the detector response (Harris score for Harris detector)
        and set the size to 10.
    '''
    image = image.astype(np.float32)
    image /= 255.
    height, width = image.shape[:2]
    features = []

    # Create grayscale image used for Harris detection
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # computeHarrisValues() computes the harris score at each pixel
    # position, storing the result in harrisImage.
    # You will need to implement this function.
    harrisImage, orientationImage = computeHarrisValues(grayImage)

    # Compute local maxima in the Harris image.  You will need to
    # implement this function. Create image to store local maximum harris
    # values as True, other pixels False
    harrisMaxImage =computeLocalMaxima(harrisImage)

    # Loop through feature points in harrisMaxImage and fill in information
    # needed for descriptor computation for each point.
    # You need to fill x, y, and angle.

    for y in range(height):
        for x in range(width):
            if not harrisMaxImage[y, x]:
                continue

            # 3: Fill in feature f with location and orientation
            # data here. Set f.size to 10, f.pt to the (x,y) coordinate,
            # f.angle to the orientation in degrees and f.response to
            # the Harris score
            f = cv2.KeyPoint()
            f.size = 10
            f.angle = orientationImage[y,x]
            f.pt = (x,y)
            f.response = harrisImage[y,x]
            features.append(f)

    return features

def Adaptive_NonMaximal_Suppression(features):
    finalfeatures = []
    n = len(features)
    m = 500

    harrisvalues = []

    for f in features:
        harrisvalues.append(f.response)

    hmax = np.max(harrisvalues)
    crobust = 0.9

    r = np.zeros(n)
    Idx = -1

    for f1 in features:
        Idx+=1

        x1,y1 = f1.pt
        if(f1.response > crobust*hmax):
            r[Idx] = float("inf")

        else:
            di = []
            for f2 in features:
                if f2.response > crobust*hmax or f2.response > crobust * f1.response:
                    continue

                if f1 == f2:
                    continue

                x2,y2 = f2.pt
                dis = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                di.append(dis)

            r[Idx] = np.max(di)

    decIdx = np.argsort(-r)

    for j in range(m):
        finalfeatures.append(features[decIdx[j]])

def ANMS(features):
    harrisvalues = []
    for f in features:
        harrisvalues.append(f.response)
    hmax = np.max(harrisvalues)

    decIdx = np.argsort(-harrisvalues)

    n = len(features)
    m = 500
    crobust = 0.9
    R = 100
    finalfeatures = []
    s = set()

    for i in range(5):
        for i in range(len(decIdx)):
            Index = decIdx[i]
            if(Index in s):
                continue

            if features[Index].response > crobust*hmax:
                finalfeatures.append(features[Index])
                continue

            flag =True

            for f2 in features:
                x1,y1 = features[Index].pt
                x2,y2 = f2.pt

                if(f2.response > crobust*features[Index] and np.sqrt((x1-x2)**2+(y1-y2)**2)<R):
                        flag = False
                        break

            if flag == True:
                finalfeatures.append(features[Index])
                s.add(Index)
                if (len(s) > m):
                    break

        if(len(s)>m):
            break
        R-=10

    return  finalfeatures




## Feature descriptors #########################################################

def describeFeatures(image, keypoints):
    '''
    Input:
        image -- BGR image with values between [0, 255]
        keypoints -- the detected features, we have to compute the feature
        descriptors at the specified coordinates
    Output:
        desc -- K x W^2 numpy array, where K is the number of keypoints
                and W is the window size
    '''
    image = image.astype(np.float32)
    image /= 255.
    # This image represents the window around the feature you need to
    # compute to store as the feature descriptor (row-major)
    windowSize = 8

    desc = np.zeros((len(keypoints), windowSize * windowSize))

    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayImage = ndimage.gaussian_filter(grayImage, 0.5)
    height,width = grayImage.shape[:2]

    newpd = np.zeros((height+40-1,width+40-1),dtype=float)
    newpd[20:20+height,20:20+width] = grayImage

    for i, f in enumerate(keypoints):
        # 5: Compute the transform as described by the feature
        # location/orientation. You will need to compute the transform
        # from each pixel in the 40x40 rotated window surrounding
        # the feature to the appropriate pixels in the 8x8 feature
        # descriptor image.
        transMx = np.zeros((2, 3))

        x,y = f.pt
        angle = -f.angle * (2*np.pi) / 360

        trans_mx1 = np.array([[1,0,-x],[0,1,-y],[0,0,1]])


        rot_mx = np.array([  [math.cos(angle), -math.sin(angle), 0],
                             [math.sin(angle), math.cos(angle), 0],
                             [0, 0, 1]])

        scale_mx = np.array([[1/5,0,0],
                             [0,1/5,0],
                             [0,0,1]])

        trans_mx2 = np.array([[1,0,4], [0,1,4], [0,0,1]])


        transMx = np.dot(trans_mx2,np.dot(scale_mx,np.dot(rot_mx,trans_mx1)))[0:2,0:3]

        # Call the warp affine function to do the mapping
        # It expects a 2x3 matrix

        destImage = cv2.warpAffine(grayImage, transMx,
            (windowSize, windowSize), flags=cv2.INTER_LINEAR)

        # Normalize the descriptor to have zero mean and unit
        # variance. If the variance is zero then set the descriptor
        # vector to zero. Lastly, write the vector to desc.

        destImage = destImage - np.mean(destImage)
        if(np.std(destImage)<10**-5):
            destImage = np.zeros((1,8*8))
        else:
            destImage = destImage / np.std(destImage)
            destImage = np.reshape(destImage, (1, 8 * 8))


        desc[i] = destImage
    return desc

def describeFeatures(image, keypoints):
    '''
    Input:
        image -- BGR image with values between [0, 255]
        keypoints -- the detected features, we have to compute the feature
        descriptors at the specified coordinates
    Output:
        Descriptor numpy array, dimensions:
            keypoint number x feature descriptor dimension
    '''

    image = image.astype(np.float32)
    image /= 255.
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    orientationImage = np.zeros(grayImage.shape[:2],dtype=float)
    desc = np.zeros((len(keypoints),16*8))
    Ix = ndimage.sobel(grayImage,axis=1,mode='reflect')
    Iy = ndimage.sobel(grayImage,axis=0,mode='reflect')
    orientationImage  = np.rad2deg(np.arctan2(Iy,Ix))


    #SIFT
    for i,f in enumerate(keypoints):
        x,y = f.pt
        x = int(x)
        y = int(y)
        contain = np.zeros((16,8))
        for outrow in range(4):
            for outcol in range(4):
                for inrow in range(4):
                    for incol in range(4):
                        distcol = outcol*4 + incol
                        distrow = outrow*4 + inrow

                        if(y-7+distrow)<0 or (y-7+distrow)>grayImage.shape[0]-1 or (x-7+distcol)<0 or (x-7+distcol)>grayImage.shape[1]-1:
                            break

                        degree = orientationImage[y-7+distrow,x-7+distcol]
                        if(degree<0):
                            degree+=360
                        degpart = int(degree//45)
                        contain[outrow*4+outcol,degpart] += 1

        contain = contain.reshape((1,128));
        stddev =np.std(contain)
        if stddev < 10**-5:
            contain = np.zeros((1,128))
        else:
            contain = (contain - np.mean(contain)) / stddev

        desc[i] = contain

    return desc


## Feature matchers ############################################################


def matchFeatures(desc1, desc2):
    '''
    Input:
        desc1 -- the feature descriptors of image 1 stored in a numpy array,
            dimensions: rows (number of key points) x
            columns (dimension of the feature descriptor)
        desc2 -- the feature descriptors of image 2 stored in a numpy array,
            dimensions: rows (number of key points) x
            columns (dimension of the feature descriptor)
    Output:
        features matches: a list of cv2.DMatch objects
            How to set attributes:
                queryIdx: The index of the feature in the first image
                trainIdx: The index of the feature in the second image
                distance: The distance between the two features
    '''
    raise NotImplementedError

# Evaluate a match using a ground truth homography.  This computes the
# average SSD distance between the matched feature points and
# the actual transformed positions.
@staticmethod
def evaluateMatch(features1, features2, matches, h):
    d = 0
    n = 0

    for m in matches:
        id1 = m.queryIdx
        id2 = m.trainIdx
        ptOld = np.array(features2[id2].pt)
        ptNew = FeatureMatcher.applyHomography(features1[id1].pt, h)

        # Euclidean distance
        d += np.linalg.norm(ptNew - ptOld)
        n += 1

    return d / n if n != 0 else 0

    # Transform point by homography.
@staticmethod
def applyHomography(pt, h):
    x, y = pt
    d = h[6]*x + h[7]*y + h[8]

    return np.array([(h[0]*x + h[1]*y + h[2]) / d,
        (h[3]*x + h[4]*y + h[5]) / d])


def matchFeatures(desc1, desc2):
    '''
    Input:
        desc1 -- the feature descriptors of image 1 stored in a numpy array,
            dimensions: rows (number of key points) x
            columns (dimension of the feature descriptor)
        desc2 -- the feature descriptors of image 2 stored in a numpy array,
            dimensions: rows (number of key points) x
            columns (dimension of the feature descriptor)
    Output:
        features matches: a list of cv2.DMatch objects
            How to set attributes:
                queryIdx: The index of the feature in the first image
                trainIdx: The index of the feature in the second image
                distance: The distance between the two features
    '''
    matches = []
    # feature count = n
    assert desc1.ndim == 2
    # feature count = m
    assert desc2.ndim == 2
    # the two features should have the type
    assert desc1.shape[1] == desc2.shape[1]

    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return []

    # Perform simple feature matching.  This uses the SSD
    # distance between two feature vectors, and matches a feature in
    # the first image with the closest feature in the second image.
    # Note: multiple features from the first image may match the same
    # feature in the second image.

    # bf = cv2.BFMatcher(cv2.NORM_L2,crossCheck = True)
    # matches = bf.match(desc1,desc2)



    # length = desc1.shape[1]
    # dist = distance.cdist(desc1,desc2,'euclidean')
    # for i in range(length):
    #     queryIdx = i
    #     imgIdx = np.argmin(dist[i,:])
    #     distan = dist[i,imgIdx]
    #     matches.append(cv2.DMatch(queryIdx,imgIdx,distan))

    # for i, desc in enumerate(desc1):
    #     dif = desc2 - desc
    #     sq = dif * dif
    #     sq = np.sum(sq,axis=1)
    #     bestInd = np.argmin(sq)
    #     match = cv2.DMatch()
    #     match.queryIdx = i
    #     match.trainIdx = bestInd
    #     match.distance = sq[bestInd]
    #     matches.append(match)

    n1 = desc1.shape[0]
    n2 = desc2.shape[0]
    distance = scipy.spatial.distance.cdist(desc1, desc2, 'euclidean')

    # print(distance)

    match = np.argmin(distance, 1)
    # print(match)
    for i in range(n1):
        f = cv2.DMatch()
        f.queryIdx = i
        f.trainIdx = int(match[i])
        f.distance = distance[i, int(match[i])]
        matches.append(f)
    return matches

def concatImages(imgs):
    imgs = [img for img in imgs if img is not None]
    maxh = max([img.shape[0] for img in imgs]) if imgs else 0
    sumw = sum([img.shape[1] for img in imgs]) if imgs else 0
    vis = np.zeros((maxh, sumw, 3), np.uint8)
    vis.fill(255)
    accumw = 0
    for img in imgs:
        h, w = img.shape[:2]
        vis[:h, accumw:accumw + w, :] = img
        accumw += w
    return vis

AQUAMARINE = (212, 255, 127)
def drawMatches(img1, kp1, img2, kp2, matches):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    vis =concatImages([img1, img2])

    kp_pairs = [[kp1[m.queryIdx], kp2[m.trainIdx]] for m in matches]
    status = np.ones(len(kp_pairs), np.bool_)
    p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.circle(vis, (x1, y1), 5, green, 2)
            cv2.circle(vis, (x2, y2), 5, green, 2)
        else:
            r = 5
            thickness = 6
            cv2.line(vis, (x1 - r, y1 - r), (x1 + r, y1 + r), red, thickness)
            cv2.line(vis, (x1 - r, y1 + r), (x1 + r, y1 - r), red, thickness)
            cv2.line(vis, (x2 - r, y2 - r), (x2 + r, y2 + r), red, thickness)
            cv2.line(vis, (x2 - r, y2 + r), (x2 + r, y2 - r), red, thickness)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), AQUAMARINE)

    return vis