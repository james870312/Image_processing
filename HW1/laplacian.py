import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

#choose read which image
img = cv.imread('Lena.bmp', cv.IMREAD_GRAYSCALE)
#img = cv.imread('Peppers.bmp', cv.IMREAD_GRAYSCALE)
#img = cv.imread('Cameraman.bmp', cv.IMREAD_GRAYSCALE)
#img = cv.imread('../HW2/blurry_moon.tif', cv.IMREAD_GRAYSCALE)
#img = cv.imread('../HW2/skeleton_orig.bmp', cv.IMREAD_GRAYSCALE)

#print height and width of input image
height, width = img.shape
print(type(img))
print(img.shape)
print("height = ", height)
print("width = ", width)

#creat new image for output & laplacian kernel
lap_s = np.zeros( (height-2, width-2), np.uint8 )
s_kernel = np.zeros((3, 3))
s_kernel[0][0] =  0
s_kernel[0][1] = -1
s_kernel[0][2] =  0
s_kernel[1][0] = -1
s_kernel[1][1] =  5
s_kernel[1][2] = -1
s_kernel[2][0] =  0
s_kernel[2][1] = -1
s_kernel[2][2] =  0

#run convolution by laplacian kernel
print("wait for convolution ......")

for i in range(height-2):
    for j in range(width-2):
        cal = (s_kernel[0][0]*img[i][j] +
               s_kernel[0][1]*img[i][j+1] +
               s_kernel[0][2]*img[i][j+2] +
               s_kernel[1][0]*img[i+1][j] +
               s_kernel[1][1]*img[i+1][j+1] +
               s_kernel[1][2]*img[i+1][j+2] +
               s_kernel[2][0]*img[i+2][j] +
               s_kernel[2][1]*img[i+2][j+1] +
               s_kernel[2][2]*img[i+2][j+2])
        if(cal > 0):lap_s[i][j] = cal
        if(cal >= 255):lap_s[i][j] = 255

print("convolution done")
cv.imshow('input', img)
cv.imshow('laplacian sharpening', lap_s)

#show histogram
plt.subplot(2, 1, 1)
plt.hist(img.ravel(), 256, [0, 255],label= 'original image')
plt.subplot(2, 1, 2)
plt.hist(lap_s.ravel(), 256, [0, 255],label= 'laplacian sharpening image')

#destroy Windows
cv.waitKey(0)
cv.destroyAllWindows()
