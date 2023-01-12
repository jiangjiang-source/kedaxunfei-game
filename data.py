import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os

goodpath='/home/jiang/下载/LED2/train/good'
badpath='/home/jiang/下载/LED2/train/bad'
testpath='/home/jiang/下载/LED2/TEST'

train_feature=[]
good=os.listdir(goodpath)
bad=os.listdir(badpath)

a,b=train_test_split(good,test_size=0.25,shuffle=True,random_state=521)


good=a



label0=[0 for i in range(len(good))]
label1=[1 for i in range(len(bad))]

train_feature=train_feature+good
train_feature=train_feature+bad
label=label0+label1

train_feature=np.array(train_feature)
label=np.array(label)

np.save('train.npy',train_feature)
np.save('label.npy',label)

