import cv2
import albumentations as A
import os
import numpy as np

#np.random.seed(42)
good=os.listdir('GOOD')
bad=os.listdir('BAD1')

D1=np.random.randint(0,len(bad),1)
D2=np.random.randint(0,len(good),1)
print(D1,D2,'BAD1/'+bad[D1[0]],'GOOD/'+good[D2[0]])
img1 = cv2.imread('BAD1/'+bad[D1[0]])
img1 = np.array(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))[150:980,350:1200]

img1=np.where(img1>20,img1,0)

A,B,C=img1.shape
IMG1=img1[:int(A/2),:int(B/2),:]
IMG2=img1[:int(A/2),int(B/2):,:]
IMG3=img1[int(A/2):,:int(B/2),:]
IMG4=img1[int(A/2):,int(B/2):,:]

IMG=(IMG1+IMG2+IMG3+IMG4)
cv2.imwrite('1.jpg',np.round(img1))
cv2.imwrite('3.jpg',IMG)


