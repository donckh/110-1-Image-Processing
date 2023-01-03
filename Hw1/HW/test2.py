import numpy as np
import cv2
import copy


def medFilter(img, imgS, filS):
    i = 0
    j = 0
    k = 0
    l = 0
    m = 0
    temp = []
    lcount = 0
    mcount = 0
    imgTempf = copy.deepcopy(img)
    for k in range(imgS[2]):
        for i in range(imgS[0]):
            for j in range(imgS[1]):
                imgTempf = conv(imgTempf, imgS, filS, temp, i, j, k)

            # print('l: ', lcount)
            # print('m: ', mcount)
    return imgTempf


def conv(imgTempff, imgSS, filSS, tempp, ii, jj, kk):
    m = 0
    l = 0
    imgNew = copy.deepcopy(imgTempff)
    for l in range(filSS[0]):
        if ii + l < imgSS[0]:
            for m in range(filSS[1]):
                if jj + m < imgSS[1]:
                    tempp.append(imgTempff[ii + l][jj + m][kk])
                else:
                    break
            m = 0
        else:
            break

    temppSum = sum(tempp)
    num = int(temppSum/len(tempp))

    if (ii + 1 < imgSS[0]) and (jj + 1 < imgSS[1]):
        print('imgTempf[', ii + 1, '][', jj + 1, '][', kk, ']: ', imgTempff[ii + 1][jj + 1][kk], ', num after: ', num, 'temppSum: ', temppSum)
        imgNew[ii + 1][jj + 1][kk] = num
    tempp.clear()

    return imgNew


image = cv2.imread('building.jpg')
imgTemp = copy.deepcopy(image)
arr = np.array(imgTemp)
imgSize = arr.shape

# print(imgTemp)
print('imSize: ', imgSize)

fNum = 3
kernel = []
for i in range(fNum):
    kernel.append([1] * fNum)

fil = np.array(kernel)
filSize = fil.shape
print('kernel: ', kernel)
print('imSize: ', filSize)

imgTemp2 = medFilter(imgTemp, imgSize, filSize)

print('imSize: ', imgSize)
print('end')

cv2.imshow('Before', image)
cv2.imshow('After', imgTemp2)
cv2.waitKey(0)

