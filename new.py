import os
import cv2

bad_path = '/home/jiang/下载/LED2/train/BAD'
a = os.listdir(bad_path)
bad_path1 = '/home/jiang/下载/LED2/train/BAD1/'
j=0
for i in a:
    img = cv2.imread(bad_path+'/'+i)[150:1000,350:1200]
    j=j+1
    address=bad_path1+str(j)+'.jpg'
    cv2.imwrite(address,img)

    img_flip = cv2.flip(img, 1)
    j = j + 1
    address = bad_path1 + str(j) + '.jpg'
    cv2.imwrite(address,img_flip)

    img_transpose = cv2.transpose(img)
    j = j + 1
    address = bad_path1 + str(j) + '.jpg'
    cv2.imwrite(address,img_transpose)

    img_flip = cv2.flip(img_transpose, 1)
    j = j + 1
    address = bad_path1 + str(j) + '.jpg'
    cv2.imwrite(address,img_flip)

    img_flip = cv2.flip(img_transpose, -1)
    j = j + 1
    address = bad_path1 + str(j) + '.jpg'
    cv2.imwrite(address,img_flip)
