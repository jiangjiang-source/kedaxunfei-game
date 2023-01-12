import cv2
import albumentations as A
import os
import numpy as np
np.random.seed(42)
def useSobel(grayImage):
    x = cv2.Sobel(grayImage, cv2.CV_16S, 1, 0)  # 对x求一阶导
    y = cv2.Sobel(grayImage, cv2.CV_16S, 0, 1)  # 对y求一阶导
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return Sobel

def useRoberts(grayImage):
    # Roberts算子
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)
    x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
    y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
    # 转uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return Roberts

def usePrewitt(grayImage):
    # Prewitt算子
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
    y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return Prewitt

def useCanny(gaussianBlur):
    #using system function
    Canny = cv2.Canny(gaussianBlur, 50, 150)
    return Canny

def useMarr(binary):
    dst = cv2.Laplacian(binary, cv2.CV_16S, ksize = 3)
    Laplacian = cv2.convertScaleAbs(dst)
    return Laplacian
l=3.5

badpath='/home/jiang/下载/LED2/train/GOOD'
bad=os.listdir(badpath)
j=0
import numpy as np

feature=[]
for i in range(len(bad)):
    address=badpath+'/'+bad[i]

    img=cv2.imread(address)
    new=np.zeros(shape=img.shape)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    TURN = img

    img =useSobel(img)
    new[:, :, 0] = (img)

    img = useSobel(TURN)
    new[:, :, 1] = img

    img = usePrewitt(TURN)
    new[:, :, 2] = img

    img=np.round(new)
    j=j+1
    address = '/home/jiang/下载/LED2/train/good/' + str(j) + '.jpg'
    cv2.imwrite(address, img*l)
# # # #############################################################

badpath='/home/jiang/下载/LED2/test'
bad=os.listdir(badpath)
j=0
for i in range(len(bad)):
    address = badpath + '/' + bad[i]

    img = cv2.imread(address)
    new = np.zeros(shape=img.shape)
    # # TURN=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[150:1000,350:1200]
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    TURN = img
    img = useSobel(TURN)
    new[:, :, 0] = img

    img = useSobel(TURN)
    new[:, :, 1] = img

    img = usePrewitt(TURN)
    new[:, :, 2] = img
    img = new

    j = j + 1
    address = '/home/jiang/下载/LED2/TEST/'+bad[i]
    cv2.imwrite(address, np.round(img)*l)



###############################################################################################################################

badpath='/home/jiang/下载/LED2/train/BAD'
bad=os.listdir(badpath)
j=0
for i in range(len(bad)):
    address = badpath + '/' + bad[i]
    if bad[i]=='196.jpg':
        print(j)
    #print(address)
    img = cv2.imread(address)
    new = np.zeros(shape=img.shape)
    # # TURN=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[150:1000,350:1200]
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    TURN = img

    img = useSobel(img)
    new[:, :, 0] = (img)

    img = useSobel(TURN)
    new[:, :, 1] = img
    img = usePrewitt(TURN)
    new[:, :, 2] = img
    img = np.round(new)


    j=j+1
    address = '/home/jiang/下载/LED2/train/bad/' + str(j) + '.jpg'
    cv2.imwrite(address, img*l)


    j = j + 1
    address = '/home/jiang/下载/LED2/train/bad/' + str(j) + '.jpg'
    img4 = l*cv2.transpose( img)
    cv2.imwrite(address, img4)


    transform3 = A.Flip(p=1)
    j = j + 1
    address = '/home/jiang/下载/LED2/train/bad/' + str(j) + '.jpg'
    img3 = l*transform3(image= img)['image']
    cv2.imwrite(address, img3)


    j = j + 1
    address = '/home/jiang/下载/LED2/train/bad/' + str(j) + '.jpg'
    img4 = cv2.transpose( img3)
    cv2.imwrite(address, img4)

    transform1 = A.HorizontalFlip(p=1)
    j=j+1
    address = '/home/jiang/下载/LED2/train/bad/' + str(j) + '.jpg'
    img1=2*transform1(image= img)['image']
    cv2.imwrite(address, img1)


    j = j + 1
    address = '/home/jiang/下载/LED2/train/bad/' + str(j) + '.jpg'
    img1=cv2.transpose( img1)
    cv2.imwrite(address, img1)

    transform2 = A.VerticalFlip(p=1)
    j = j + 1
    address = '/home/jiang/下载/LED2/train/bad/' + str(j) + '.jpg'
    img2 = l*transform2(image= img)['image']
    cv2.imwrite(address, img2)



    j = j + 1
    address = '/home/jiang/下载/LED2/train/bad/' + str(j) + '.jpg'
    img2 = cv2.transpose( img2)
    cv2.imwrite(address, img2)