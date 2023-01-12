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
# img1 = cv2.imread('BAD1/'+bad[D1[0]])
# img1 = np.array(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
# cv2.imwrite('1.jpg',np.array(img1))
#
# img2= cv2.imread('GOOD/'+good[D2[0]])
# img2 = np.array(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
# cv2.imwrite('2.jpg',np.array(img2))
#
# img=(img1*0.75+img2+0.25)
# cv2.imwrite('3.jpg',np.array(img))

l=50
for i in range(len(good)):
    #img1 = cv2.imread('BAD1/' + bad[i])
    img1 = cv2.imread('GOOD/' + good[i])
    img1 = np.array(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    img1 = np.where(img1>20, img1, 0)

    img2 = cv2.imread('GOOD/' + good[122])
    address = '/home/jiang/下载/LED2/train/BAD1/' + str(l) + '.jpg'
    img = (3*img2 - 2*img1)
    cv2.imwrite(address, np.array(img))
    l+=1

    # D = []
    # for j in range(5):
    #     D2 = np.random.randint(0, len(good), 1)
    #     if D2[0] in D:
    #         continue
    #     D.append(D2[0])
    #     img2 = cv2.imread('GOOD/' + good[D2[0]])
    #     img2 = np.array(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    #     img2= np.where(img2 > 30, img1, 0)
    #     address='/home/jiang/下载/LED2/train/BAD1/'+str(l)+'.jpg'
    #     img = (img2-img1)
    #     cv2.imwrite(address,np.array(img))
    #     l=l+1

# for i in range(len(bad)):
#     img1 = cv2.imread('BAD1/' + bad[i])
#     img1 = np.array(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
#     D = []
#     for j in range(5):
#         D2 = np.random.randint(0, len(good), 1)
#         if D2[0] in D:
#             continue
#         D.append(D2[0])
#         img2 = cv2.imread('GOOD/' + good[D2[0]])
#         img2 = np.array(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
#         address='/home/jiang/下载/LED2/train/BAD1/'+str(l)+'.jpg'
#         img = (img1 * 0.9 + img2 + 0.15)
#         cv2.imwrite(address,np.array(img))
#         l=l+1



