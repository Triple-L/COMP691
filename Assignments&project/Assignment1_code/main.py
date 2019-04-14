'''Created by Jemma for  CV class @ Concordia Univ 2019/01/15
'''

import cv2
import numpy as np

from utlis import *
import math

img = cv2.imread('oldwell_mosaic.bmp',0)
img_orignal=cv2.imread('oldwell.jpg',1)
# cv2.imshow('crayons',img_orignal)
# cv2.waitKey(0)

##########Part1#############


R,G,B = splitRGB(img)

R_filtered,G_filtered,B_filtered = AppliedFilter(R,G,B)

img_f=cv2.merge([B_filtered,G_filtered,R_filtered])

diff = CmpDiffrence(img_orignal,img_f)

#Show result Images
cv2.imshow('Part1:img_orignal',img_orignal)
cv2.imshow('Part1:img_filterd',img_f)
cv2.imshow('Part1:diff',diff)
# cv2.waitKey(0)
#
# cv2.imwrite('Part1_img_orignal.jpg',img_orignal)
# cv2.imwrite('Part1_img_filterd.jpg',img_f)
# cv2.imwrite('Part1_diff.jpg',diff)

###############Part2##############

B_R = median(B_filtered,R_filtered)
G_R = median(G_filtered,R_filtered)

IMG = cv2.merge([B_R,G_R,R_filtered])

diff2 = CmpDiffrence(img_orignal,IMG)


cv2.imshow('Part2_img_orignal',img_orignal)
cv2.imshow('Part2_demosaiced_img',IMG)
cv2.imshow('Part2_diff2',diff2)

# cv2.imwrite('Part2_img_orignal.jpg',img_orignal)
# cv2.imwrite('Part2_demosaiced_img.jpg',IMG)
# cv2.imwrite('Part2_diff2.jpg',diff2)
cv2.waitKey(0)










