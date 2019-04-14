#This works! For 6 pics， self take pics and Hanging pics
import cv2
from utils import *

img1 = cv2.imread('./project_images/Rainier1.png')# queryImage
img2 = cv2.imread('./project_images/Rainier2.png')

# M_54 = Stitch_function(img1, img2)
# cv2.imshow("f",M_54)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img3 = cv2.imread('./project_images/Rainier3.png')
img4 = cv2.imread('./project_images/Rainier4.png')
img5 = cv2.imread('./project_images/Rainier5.png')
img6 = cv2.imread('./project_images/Rainier6.png')

M_54 = Stitch_function(img5, img4)
cv2.imwrite("M_54.png",M_54)
M_54 = cv2.imread("M_54.png")
M_ = Stitch_function(M_54, img6)
M_ = Stitch_function(M_, img1)
cv2.imwrite("M_546.png",M_)
M_ = Stitch_function(M_, img2)
cv2.imwrite("M_5462.png",M_)
M_ = Stitch_function(M_, img3)
cv2.imwrite("M_54623.png",M_)

cv2.imshow("f",M_)
cv2.waitKey(0)
cv2.destroyAllWindows()

