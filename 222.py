import cv2
import albumentations as A
import os
import numpy as np
np.random.seed(42)

import pandas as pd
train_feature=[]
weilabel=pd.read_csv('/home/jiang/桌面/game1/sub1.csv')

for name in weilabel['image']:
   # print(name)
    name1='/home/jiang/下载/LED2/test/'+name
    s=weilabel[weilabel['image']==name]
    if s['label'].values[0]==1:
        train_feature.append(name1)
j=0
for i in train_feature:
    img=cv2.imread(i)
   # print(i[24:])

    j=j+1
    if j==241:
         address = '/home/jiang/下载/LED/1/' + i[25:]
         print(address)
    address = '/home/jiang/下载/LED/1/' + str(j)+'.jpg'
  #  address = '/home/jiang/下载/LED2/1/' + i[25:]
    cv2.imwrite(address, img)