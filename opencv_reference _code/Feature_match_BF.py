# Use the Brute-Force matcher and FLANN Matcher in OpenCV
#This works Features are ORB

import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('./project_images/Rainier1.png',0) # queryImage
orb1 = cv2.ORB_create()# Initiate SIFT detector
# find the keypoints and descriptors with SIFT
kp1, des1 = orb1.detectAndCompute(img1, None)

img2 = cv2.imread('./project_images/Rainier2.png',0) # queryImage
orb2 = cv2.ORB_create()
# Initiate SIFT detector

# find the keypoints and descriptors with SIFT
kp2, des2 = orb2.detectAndCompute(img2, None)

# creaimplement Project(x1,y1,H,x2,y2) te BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

img3 = img1
# Draw first matches.
img3 =cv2.drawMatches(img1,kp1,img2,kp2,matches[:],img3)
cv2.imshow("img3",img3)
cv2.waitKey(0)

# src_pts = np.float32([m.pt for m in kp1]).reshape(-1, 1, 2)
# dst_pts = np.float32([j.pt for j in kp2]).reshape(-1, 1, 2)
# H2 = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC)
# print(H2)
