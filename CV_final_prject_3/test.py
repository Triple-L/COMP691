#%pylab inline
import cv2
import numpy as np


detector = cv2.xfeatures2d.GFTT()


    # cv2.FeatureDetector_create("GFTT")
descriptor = cv2.xfeatures2d.SURF_create(400)


img = cv2.imread('1a.png',0)
# detect keypoints
kp1 = detector.detect(img)
kp2 = detector.detect(img)

print('#keypoints in image1: %d, image2: %d' % (len(kp1), len(kp2)))

