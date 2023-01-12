import numpy as np
import pandas as pd
from sklearn.utils import shuffle
data1=pd.read_csv('train.csv')
data2=pd.read_csv('/home/jiang/下载/科大讯飞/LED_part2_train/train.csv')

data1['address']=data1[' file label'].apply(lambda x:x.split(' ')[1])#.astype('int')
data1['label']=data1[' file label'].apply(lambda x:x.split(' ')[2]).astype('int')

data2['address']=data2[' file label'].apply(lambda x:x.split(' ')[1])#.astype('int')
data2['label']=data2[' file label'].apply(lambda x:x.split(' ')[2]).astype('int')


data11=data1[data1['label']==1]
data12=data2[data2['label']==1]
data21=data1[data1['label']==2]
data22=data2[data2['label']==2]
data31=data1[data1['label']==3]
data32=data2[data2['label']==3]

m1= min(len(data11),len(data12))
m2= min(len(data11),len(data12))
m3= min(len(data11),len(data12))
print(len(data1))
import cv2

w1=0.75
w2=0.25
for i in range(m1):
    for j in range(10):
        s1=data11.values[j,1]
        s2= data12.values[i, 1]
        name='ff'
        train_feature = '/home/jiang/下载/科大讯飞/LED_part1_train/imgs/' + s1+ '.bmp'
        test_feature = '/home/jiang/下载/科大讯飞/LED_part2_train/imgs/' + s2 + '.bmp'
        img11=cv2.resize(cv2.imread(train_feature),(512,160))
        img12=cv2.resize(cv2.imread(test_feature),(512,160))

        img=w1*img11+w2*img12
        name='/home/jiang/下载/科大讯飞/LED_part1_train/imgs/new1'+str(i)+str(j)+'.bmp'
        cv2.imwrite(name,img)
        d='1'+' '+'new1'+str(i)+str(j)+' '+'1'
        new=pd.DataFrame([[d,1,1]],columns=[' file label','address','label'])
        data1=pd.concat((data1,new))

for i in range(m2):
    for j in range(10):
        s1 = data21.values[j, 1]
        s2 = data22.values[i, 1]
        name = 'ff'
        train_feature = '/home/jiang/下载/科大讯飞/LED_part1_train/imgs/' + s1 + '.bmp'
        test_feature = '/home/jiang/下载/科大讯飞/LED_part2_train/imgs/' + s2 + '.bmp'
        img11 = cv2.resize(cv2.imread(train_feature), (512, 160))
        img12 = cv2.resize(cv2.imread(test_feature), (512, 160))
        img=w1*img11+w2*img12
        name='/home/jiang/下载/科大讯飞/LED_part1_train/imgs/new2'+str(i)+str(j)+'.bmp'
        cv2.imwrite(name,img)
        d='1'+' '+'new2'+str(i)+str(j)+' '+'2'
        new=pd.DataFrame([[d,1,1]],columns=[' file label','address','label'])
        data1=pd.concat((data1,new))

for i in range(m3):
    for j in range(10):
        s1 = data31.values[j, 1]
        s2 = data32.values[i, 1]
        name = 'ff'
        train_feature = '/home/jiang/下载/科大讯飞/LED_part1_train/imgs/' + s1 + '.bmp'
        test_feature = '/home/jiang/下载/科大讯飞/LED_part2_train/imgs/' + s2 + '.bmp'
        img11 = cv2.resize(cv2.imread(train_feature), (512, 160))
        img12 = cv2.resize(cv2.imread(test_feature), (512, 160))
        img=w1*img11+w2*img12
        name='/home/jiang/下载/科大讯飞/LED_part1_train/imgs/new3'+str(i)+str(j)+'.bmp'
        cv2.imwrite(name,img)
        d='1'+' '+'new3'+str(i)+str(j)+' '+'3'
        new=pd.DataFrame([[d,1,1]],columns=[' file label','address','label'])
        data1=pd.concat((data1,new))
data1[' file label'].to_csv('train1.csv',index=None)
print(data1.tail())