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

def computeHarrisValues(srcImage):

    height, width = srcImage.shape[:2]

    harrisImage = np.zeros(srcImage.shape[:2],dtype=float)
    orientationImage = np.zeros(srcImage.shape[:2],dtype=float)

    sobx = np.zeros(srcImage.shape[:2],dtype=float)
    filters.sobel(srcImage,1,sobx)
    soby = np.zeros(srcImage.shape[:2],dtype=float)
    filters.sobel(srcImage,0,soby)

    Ix = sobx*sobx
    Iy = soby*soby
    Ixy = sobx*soby


    Wxx = filters.gaussian_filter(Ix,sigma=0.5)
    Wyy = filters.gaussian_filter(Iy,sigma=0.5)
    Wxy = filters.gaussian_filter(Ixy,sigma=0.5)

    harrisImage = (Wxx*Wyy - Wxy*Wxy)/(Wxx+Wyy)
    orientationImage  = np.arctan2(soby,sobx)*(180) / np.pi
    return harrisImage, orientationImage

def checkBorder( va, borderA, vb, borderB):
        if va - 1 >= 0 and va + 1 < borderA and vb - 1 >= 0 and vb + 1 < borderB:
            return True
        else:
            return False

def computeLocalMaxima(harrisImage):

    height, width = harrisImage.shape[:2]
    destImage = np.zeros_like(harrisImage, np.bool)
    # 2: Compute the local maxima image
    newmax = ndimage.maximum_filter(harrisImage,size=3)
    for i in range(height):
        for j in range(width):
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

    harrisMaxImage =computeLocalMaxima(harrisImage)

    for y in range(height):
        for x in range(width):
            if not harrisMaxImage[y, x]:
                continue

            f = cv2.KeyPoint()
            f.size = 10
            f.angle = orientationImage[y,x]
            f.pt = (x,y)
            f.response = harrisImage[y,x]
            features.append(f)
    return features

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




## Feature descriptors #######

def describeFeatures(image, keypoints):

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


## Feature matchers #######

def evaluateMatch(features1, features2, matches, h):
    d = 0
    n = 0

    for m in matches:
        id1 = m.queryIdx
        id2 = m.trainIdx
        ptOld = np.array(features2[id2].pt)
        ptNew = applyHomography(features1[id1].pt, h)

        # Euclidean distance
        d += np.linalg.norm(ptNew - ptOld)
        n += 1

    return d / n if n != 0 else 0

def applyHomography(pt, h):
    x, y = pt
    d = h[6]*x + h[7]*y + h[8]

    return np.array([(h[0]*x + h[1]*y + h[2]) / d,
        (h[3]*x + h[4]*y + h[5]) / d])

def matchFeatures(desc1, desc2):

    matches = []
    assert desc1.ndim == 2
    assert desc2.ndim == 2
    assert desc1.shape[1] == desc2.shape[1]

    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return []

    n1 = desc1.shape[0]
    n2 = desc2.shape[0]
    distance = scipy.spatial.distance.cdist(desc1, desc2, 'euclidean')

    match = np.argmin(distance, 1)

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

def drawMatches(img1, kp1, img2, kp2, matches):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    Matched =concatImages([img1, img2])

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
            cv2.circle(Matched, (x1, y1), 5, green, 2)
            cv2.circle(Matched, (x2, y2), 5, green, 2)
        else:
            r = 5
            thickness = 6
            cv2.line(Matched, (x1 - r, y1 - r), (x1 + r, y1 + r), red, thickness)
            cv2.line(Matched, (x1 - r, y1 + r), (x1 + r, y1 - r), red, thickness)
            cv2.line(Matched, (x2 - r, y2 - r), (x2 + r, y2 + r), red, thickness)
            cv2.line(Matched, (x2 - r, y2 + r), (x2 + r, y2 - r), red, thickness)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(Matched, (x1, y1), (x2, y2), (212, 255, 127))

    return Matched