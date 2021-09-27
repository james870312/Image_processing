import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

#choose read which image
img = cv.imread('image1.jpg', cv.IMREAD_GRAYSCALE)
#img = cv.imread('image2.jpg', cv.IMREAD_GRAYSCALE)
#img = cv.imread('image3.jpg', cv.IMREAD_GRAYSCALE)

#print height and width of input image
height, width = img.shape
print(type(img.shape))
print(img.shape)
print("height = ", height)
print("width = ", width)
print("*************")

#creat new image for output & laplacian kernel
lap_s = np.zeros( (height-2, width-2), np.float )
l_kernel = np.array([[ 0.0,-1.0, 0.0],
                     [-1.0, 4.0,-1.0],
                     [ 0.0,-1.0, 0.0]])

#run convolution by laplacian kernel
print("wait for convolution ......")

for i in range(height-2):
    for j in range(width-2):
            lap_s[i, j] = np.sum(l_kernel * img[i:i+3, j:j+3])
            if(lap_s[i, j] < 0):lap_s[i, j] = 0
            if(lap_s[i, j] >= 255):lap_s[i, j] = 255

print("convolution done")
cv.imshow('source', img)
cv.imshow('Laplacian edge detection', lap_s.astype(np.uint8))

#destroy Windows
cv.waitKey(0)
cv.destroyAllWindows()
