import cv2
import albumentations as A
import os
import numpy as np
np.random.seed(42)
badpath='/home/jiang/下载/LED2/train/BAD'
badpath1='/home/jiang/下载/LED2/train/BAD1/'
bad=os.listdir(badpath)
j=0



j=-1
for i in bad:
    address=badpath+'/'+i
    img = cv2.imread(address)
    img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    j=j+1
    newaddress=badpath1+str(j)+'.jpg'
    cv2.imwrite(newaddress,img)

    # j = j + 1
    # newaddress = badpath1 + str(j) + '.jpg'
    #
    # a=0.1
    # img = cv2.imread(address)
    # img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # img[:,:,0]=img[:,:,0]*a
    # img[:, :, 1] = img[:, :, 1] *a
    # img[:, :, 2] = img[:, :, 2] * a
    # cv2.imwrite(newaddress, np.round(img))

    j = j + 1
    newaddress = badpath1 + str(j) + '.jpg'

    a = 0.2
    img = cv2.imread(address)
    img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img[:, :, 0] = img[:, :, 0] * a
    img[:, :, 1] = img[:, :, 1] * a
    img[:, :, 2] = img[:, :, 2] * a
    cv2.imwrite(newaddress, np.round(img))

    # j = j + 1
    # newaddress = badpath1 + str(j) + '.jpg'
    #
    # a=0.3
    # img = cv2.imread(address)
    # img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # img[:,:,0]=img[:,:,0]*a
    # img[:, :, 1] = img[:, :, 1] *a
    # img[:, :, 2] = img[:, :, 2] * a
    # cv2.imwrite(newaddress, np.round(img))

    j = j + 1
    newaddress = badpath1 + str(j) + '.jpg'

    a = 0.4
    img = cv2.imread(address)
    img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img[:, :, 0] = img[:, :, 0] * a
    img[:, :, 1] = img[:, :, 1] * a
    img[:, :, 2] = img[:, :, 2] * a
    cv2.imwrite(newaddress, np.round(img))
    #
    # j = j + 1
    # newaddress = badpath1 + str(j) + '.jpg'


    # a=0.5
    # img = cv2.imread(address)
    # img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # img[:,:,0]=img[:,:,0]*a
    # img[:, :, 1] = img[:, :, 1] *a
    # img[:, :, 2] = img[:, :, 2] * a
    # cv2.imwrite(newaddress, np.round(img))
    #
    # j = j + 1
    # newaddress = badpath1 + str(j) + '.jpg'

    # a=0.6
    # img = cv2.imread(address)
    # img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # img[:,:,0]=img[:,:,0]*a
    # img[:, :, 1] = img[:, :, 1] *a
    # img[:, :, 2] = img[:, :, 2] * a
    # cv2.imwrite(newaddress, np.round(img))

    j = j + 1
    newaddress = badpath1 + str(j) + '.jpg'

    a=0.7
    img = cv2.imread(address)
    img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img[:,:,0]=img[:,:,0]*a
    img[:, :, 1] = img[:, :, 1] *a
    img[:, :, 2] = img[:, :, 2] * a
    cv2.imwrite(newaddress, np.round(img))

    j = j + 1
    newaddress = badpath1 + str(j) + '.jpg'

    a=0.9
    img = cv2.imread(address)
    img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img[:,:,0]=img[:,:,0]*a
    img[:, :, 1] = img[:, :, 1] *a
    img[:, :, 2] = img[:, :, 2] * a
    cv2.imwrite(newaddress, np.round(img))

    j = j + 1
    newaddress = badpath1 + str(j) + '.jpg'

    # a=1.1
    # img = cv2.imread(address)
    # img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # img[:,:,0]=img[:,:,0]*a
    # img[:, :, 1] = img[:, :, 1] *a
    # img[:, :, 2] = img[:, :, 2] * a
    # cv2.imwrite(newaddress, np.round(img))
    #
    # j = j + 1
    # newaddress = badpath1 + str(j) + '.jpg'

    a=1.3
    img = cv2.imread(address)
    img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img[:,:,0]=img[:,:,0]*a
    img[:, :, 1] = img[:, :, 1] *a
    img[:, :, 2] = img[:, :, 2] * a
    cv2.imwrite(newaddress, np.round(img))

    # j = j + 1
    # newaddress = badpath1 + str(j) + '.jpg'
    #
    # # a=1.5
    # # img = cv2.imread(address)
    # # img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # # img[:,:,0]=img[:,:,0]*a
    # # img[:, :, 1] = img[:, :, 1] *a
    # # img[:, :, 2] = img[:, :, 2] * a
    # # cv2.imwrite(newaddress, np.round(img))
    #
    # j = j + 1
    # newaddress = badpath1 + str(j) + '.jpg'
    #
    # a=1.7
    # img = cv2.imread(address)
    # img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # img[:,:,0]=img[:,:,0]*a
    # img[:, :, 1] = img[:, :, 1] *a
    # img[:, :, 2] = img[:, :, 2] * a
    # cv2.imwrite(newaddress, np.round(img))
    #
    #
    # j = j + 1
    # img = cv2.imread(address)
    # img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # newaddress = badpath1 + str(j) + '.jpg'


    # a=1.9
    # img = cv2.imread(address)
    # img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # img[:,:,0]=img[:,:,0]*a
    # img[:, :, 1] = img[:, :, 1] *a
    # img[:, :, 2] = img[:, :, 2] * a
    # cv2.imwrite(newaddress, np.round(img))

    # a=2.1
    # img = cv2.imread(address)
    # img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # img[:,:,0]=img[:,:,0]*a
    # img[:, :, 1] = img[:, :, 1] *a
    # img[:, :, 2] = img[:, :, 2] * a
    # cv2.imwrite(newaddress, np.round(img))
    #
    # j = j + 1
    # newaddress = badpath1 + str(j) + '.jpg'
    #
    # a=2.3
    # img = cv2.imread(address)
    # img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # img[:,:,0]=img[:,:,0]*a
    # img[:, :, 1] = img[:, :, 1] *a
    # img[:, :, 2] = img[:, :, 2] * a
    # cv2.imwrite(newaddress, np.round(img))
    #
    # j = j + 1
    # newaddress = badpath1 + str(j) + '.jpg'