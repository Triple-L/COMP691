#Hanging photos
import cv2
from utils import *

LB1=cv2.imread('./project_images/Hanging1.png')
LB2=cv2.imread('./project_images/Hanging2.png')
# LB2=cv2.imread('./project_images/LB/WechatIMG270.jpg')

dim = (640,480)
LB1=cv2.resize(LB1,dim)
LB2=cv2.resize(LB2,dim)
# LB3=cv2.resize(LB3,dim)
M_LB=Stitch_function(LB1, LB2,'MOPS')
# M_LBH=Stitch_function(M_LB, LB3)
cv2.imshow("img_final",M_LB)
cv2.resizeWindow("img_final",dim)
cv2.waitKey(0)
cv2.destroyAllWindows()