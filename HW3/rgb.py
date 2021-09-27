import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

#choose read which image
img = cv.imread('aloe.jpg', cv.IMREAD_COLOR)
#img = cv.imread('church.jpg', cv.IMREAD_COLOR)
#img = cv.imread('kitchen.jpg', cv.IMREAD_COLOR)
#img = cv.imread('house.jpg', cv.IMREAD_COLOR)

#print height and width of input image
height, width, channel = img.shape
print(type(img.shape))
print(img.shape)
print("height = ", height)
print("width = ", width)
print("channel = ", channel)
print("*************")

#creat new image for output & laplacian kernel
lap_s = np.zeros( (height-2, width-2, channel), np.float )
l_kernel = np.array([[ 0.0,-1.0, 0.0],
                     [-1.0, 5.0,-1.0],
                     [ 0.0,-1.0, 0.0]])

#run convolution by laplacian kernel
print("wait for convolution ......")

for i in range(height-2):
    for j in range(width-2):
        for k in range(channel):
            lap_s[i, j, k] = np.sum(l_kernel * img[i:i+3, j:j+3, k])
            if(lap_s[i, j, k] < 0):lap_s[i, j, k] = 0
            if(lap_s[i, j, k] >= 255):lap_s[i, j, k] = 255

print("convolution done")
cv.imshow('source', img)
#cv.imshow('laplacian_sharpening(float)', lap_s)
cv.imshow('laplacian_sharpening(uint8)', lap_s.astype(np.uint8))

#show histogram
plt.subplot(2, 1, 1)
plt.hist(img.ravel(), 256, [0, 255],label= 'original image')
plt.subplot(2, 1, 2)
plt.hist(lap_s.ravel(), 256, [0, 255],label= 'laplacian sharpening image')

#destroy Windows
cv.waitKey(0)
cv.destroyAllWindows()
