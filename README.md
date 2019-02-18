## Notes for COMP691_2019 winter 


### Quiz1(2019-02-12):

1. 20个选择题
2. 4个简答
    * 计算interger image 
    * DOF and corresponing points 
    * Hough transfer 伪代码 
    * Bag of feature
3. 2个大题

    * 描述Harris corner detection过程 
    * 解释RANSAC过程

### Assignment2

1. Harris corner detection
input:Imgae
output: dst_array the C value of each pixel

Using the Non_maxium suppresion to keep the max interest points in 3*3 kernel.

For Non-maxium suppression:

2. descriptor

3. Macthing

import cv2
import features
from pylab import *
from numpy import *
from PIL import Image
from features import *


image1 = np.array(Image.open('./yosemite/yosemite1.jpg'))
image2 = np.array(Image.open('./yosemite/yosemite2.jpg'))

features1=detectKeypoints(image1)
features2 =detectKeypoints(image2)

Describe1=describeFeatures(image1,features1)
Describe2=describeFeatures(image2,features2)

matches=matchFeatures(Describe1,Describe2)
matches = sorted(matches, key=lambda x: x.distance)
img3 = drawMatches(image1,features1,image2,features2,matches[100:200])
plt.imshow(img3),plt.show()'''
'''


