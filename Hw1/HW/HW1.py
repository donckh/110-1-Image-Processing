import numpy as np
import cv2
import copy

def CountPixel(original_image, ArrayB, ArrayG, ArrayR, imageH, imageW):
    B=0
    G=1
    R=2
    for i in range(imageH):
        for j in range(imageW):
            ArrayB[original_image[i][j][B]] += 1
            ArrayG[original_image[i][j][G]] += 1
            ArrayR[original_image[i][j][R]] += 1

    return ArrayB, ArrayG, ArrayR

def Accumlate(Array):
    AccumArray = [0]*256
    j = 0
    for i in range(256):
        AccumArray[i] = Array[i] + j
        j = AccumArray[i]

    return AccumArray


def ImageHeq(original_image, ACPAB, ACPAG, ACPAR, imageH, imageW):
    img_tmp = original_image
    R = 2
    G = 1
    B = 0
    for i in range(imageH):
        for j in range(imageW):
            img_tmp[i][j][B] = ACPAB[img_tmp[i][j][B]] * 255
            img_tmp[i][j][G] = ACPAG[img_tmp[i][j][G]] * 255
            img_tmp[i][j][R] = ACPAR[img_tmp[i][j][R]] * 255

    return img_tmp


image = cv2.imread('field.jpg')
image_temp = copy.deepcopy(image)
arr = np.array(image)
imageSize = arr.shape

H=imageSize[0]
W=imageSize[1]
IRed = [0]*256
IGreen = [0]*256
IBlue = [0]*256
AcPAR = [0]*256
AcPAG = [0]*256
AcPAB = [0]*256
X = range(256)

CountPixel(image_temp, IBlue, IGreen, IRed, imageSize[0], imageSize[1])
print('OriRed:', IRed,
      '\nOriGreen:', IGreen,
      '\nOriBlue:', IBlue)

EveryPAR = np.array(IRed)/(H*W)
EveryPAG = np.array(IGreen)/(H*W)
EveryPAB = np.array(IBlue)/(H*W)

print('Red:', EveryPAR,
      '\nGreen:', EveryPAG,
      '\nBlue:', EveryPAB)

AcPAR = Accumlate(EveryPAR)
AcPAG = Accumlate(EveryPAG)
AcPAB = Accumlate(EveryPAB)


print('AcRed:', AcPAR,
      '\nAcGreen:', AcPAG,
      '\nAcBlue:', AcPAB)

image_temp = ImageHeq(image_temp, AcPAB, AcPAG, AcPAR, imageSize[0], imageSize[1])


cv2.imshow('Before', image)
cv2.imshow('After', image_temp)
cv2.waitKey(0)
print('end')