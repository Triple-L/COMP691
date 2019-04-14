from pylab import *

from PIL import Image
from features import *

image1 = np.array(Image.open('./yosemite/yosemite1.jpg'))
image2 = np.array(Image.open('./yosemite/yosemite2.jpg'))

features1=detectKeypoints(image1)
features2 =detectKeypoints(image2)

Describe1=describeFeatures(image1,features1)
Describe2=describeFeatures(image2,features2)

matches=matchFeatures(Describe1,Describe2)
matches = sorted(matches, key=lambda x: x.distance)[100:200]
img3 = drawMatches(image1,features1,image2,features2,matches)
plt.imshow(img3),plt.show()


