import numpy as np
import cv2
import copy
'''
a = np.array([[1, 2], [3, 4]])
print('a :',a)
col = [8, 9]
np.insert(a, a.shape[1], col, axis=1)
print('a After:',a)

a = np.array([1, 2 ,3])
b = np.array([4, 5, 6])
c = np.vstack((a,b))


print('c After:',c)
'''
c = np.array([[[1, 2, 3],
     [2, 4, 6],
     [3, 6, 9]],
     [[4, 8, 12],
     [5, 10, 15],
     [6, 12, 18]],
     [[7, 14, 21],
     [8, 16, 24],
     [9, 18, 27]]])

d=c[:][:][0]
arrD = np.array(d)
dSize = arrD.shape

arrC = np.array(c)
cSize = arrC.shape
print('cbefore: ', c)
print('cSizebefore: ', cSize)
print('dbefore: ', d)
print('dSizebefore: ', dSize)

d = np.vstack(d, 1, [0, 0, 0], axis=1)
arrD = np.array(d)
dSize = arrD.shape

print('d: ', d)
print('dSize: ', dSize)

'''a=[0,1,2,3,4,5,6,7,8]
b=len(a)/2

print('len:', len(a), '\n/2: ', b, '\n/2+1: ', int(b+1), '\n/2+1: ', a[int(b+1)])

image = cv2.imread('building.jpg')
imTemp = copy.deepcopy(image)
arr = np.array(imTemp)
imSize = arr.shape
img = cv2.cvtColor(imTemp,cv2.COLOR_RGB2GRAY)
arr2 = np.array(imTemp)
imSize2 = arr2.shape

print(imTemp)
print('imSize: ', imSize)
print(img)
print('imSize2: ', imSize2)


filter = []
fNum = 3
for i in range(fNum):
    filter.append([1] * fNum)

fil = np.array(filter)
filSize = fil.shape
print('kernel: ', filter)
print('imSize: ', filSize)

cv2.imshow('Before', imTemp)
cv2.imshow('After', img)
cv2.waitKey(0)'''