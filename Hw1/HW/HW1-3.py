import numpy as np
import cv2
import copy

def medFilter(img, imgS, ker, filS):
    i = 0
    j = 0
    l = 0
    m = 0
    temp = []
    lcount = 0
    mcount = 0
    imgTempf = copy.deepcopy(img)
    for i in range(imgS[0]):
        for j in range(imgS[1]):
            imgTempf = conv(imgTempf, imgS, ker, filS, temp, i, j)

            # print('l: ', lcount)
            # print('m: ', mcount)
    return imgTempf


def conv(imgTempff, imgSS, ker, filSS, tempp, ii, jj):
    m = 0
    l = 0
    imgNew = copy.deepcopy(imgTempff)
    imgNew2 = copy.deepcopy(imgTempff)
    for l in range(filSS[0]):
        if ii + l < imgSS[0]:
            for m in range(filSS[1]):
                if jj + m < imgSS[1]:
                    tempp.append(imgNew2[ii + l][jj + m] * ker[l][m])
                    #print('imgTempff: [', ii + l, '], [', jj + m, '] ', imgTempff[ii + l][jj + m], 'ker[', l, '][', m, ']: ', ker[l][m], 'tempp:', tempp)
                else:
                    break
            m = 0
        else:
            break

    temppSum = sum(tempp)
    num = int(temppSum / (filSS[0] * filSS[1]))

    if (ii + 1 < imgSS[0]) and (jj + 1 < imgSS[1]) and (len(tempp) >= filSS[0] * filSS[1]):
        #print('imgTempf[', ii + 1, '][', jj + 1, ']', imgTempff[ii + 1][jj + 1], 'num: ', num, 'tempp: ', tempp, 'temppSum: ', temppSum)
        if temppSum >= 255:
            imgNew[ii + 1][jj + 1] = 255
        elif temppSum >= 0:
            imgNew[ii + 1][jj + 1] = temppSum
        else:
             imgNew[ii + 1][jj + 1] = 0
    tempp.clear()

    return imgNew


image = cv2.imread('test2.png')
imgTemp = copy.deepcopy(image)
imgGray = cv2.cvtColor(imgTemp, cv2.COLOR_RGB2GRAY)
arr = np.array(imgGray)
imgSize = arr.shape

# print(imgTemp)
print('imSize: ', imgSize)
for z in range(100):
    for x in range(100):
        if (imgGray[z][x] !=255) and imgGray[z][x] !=0:
            print('image[', z, ']', '[', x, ']', imgGray[z][x])

# fNum = 3
kernel = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
#kernel = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
# for i in range(fNum):
#     kernel.append([1] * fNum)

fil = np.array(kernel)
filSize = fil.shape
print('kernel: ', kernel)
print('imSize: ', filSize)

imgTemp2 = medFilter(imgGray, imgSize, kernel, filSize)

print('imSize: ', imgSize)
print('end')

cv2.imshow('Before', imgGray)
cv2.imshow('After', imgTemp2)
cv2.waitKey(0)

