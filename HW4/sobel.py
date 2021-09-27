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

#set parameter
sigma = 1.0;
kernel = 3;
gauss = np.zeros( (height-2, width-2), np.float )
Gx = np.zeros( (height-4, width-4), np.float )
Gy = np.zeros( (height-4, width-4), np.float )
sobel = np.zeros( (height-4, width-4), np.float )
G_kernel = np.zeros( (kernel, kernel), np.float )

Gx_kernel = np.array([[-1.0, 0.0, 1.0],
                      [-2.0, 0.0, 2.0],
                      [-1.0, 0.0, 1.0]])

Gy_kernel = np.array([[-1.0,-2.0,-1.0],
                      [ 0.0, 0.0, 0.0],
                      [ 1.0, 2.0, 1.0]])

#Create gaussian kernel
for i in range(kernel):
    for j in range(kernel):
        G_kernel[i,j] =math.exp(-1*((i-int(kernel/2))**2 + (j-int(kernel/2))**2)/(2*sigma**2))

#blur image by gaussian kernel
print("wait for convolution ......")
for i in range(height-2):
    for j in range(width-2):
            gauss[i, j] = np.sum(G_kernel * img[i:i+3, j:j+3])/np.sum(G_kernel)
            
#detect edge by sobel kernel
for i in range(height-4):
    for j in range(width-4):
            Gx[i, j] = np.sum(Gx_kernel * gauss[i:i+3, j:j+3])
            if(Gx[i, j] < 0):Gx[i, j] = 0
            elif(Gx[i, j] >= 255):Gx[i, j] = 255
            Gy[i, j] = np.sum(Gy_kernel * gauss[i:i+3, j:j+3])
            if(Gy[i, j] < 0):Gy[i, j] = 0
            elif(Gy[i, j] >= 255):Gy[i, j] = 255
            sobel[i, j] = math.sqrt(Gx[i, j]**2 + Gy[i, j]**2)
            if(sobel[i, j] >= 255):sobel[i, j] = 255

print("convolution done")
#show image
cv.imshow('source', img)
cv.imshow('Gauss', gauss.astype(np.uint8))
cv.imshow('Gx', Gx.astype(np.uint8))
cv.imshow('Gy', Gy.astype(np.uint8))
cv.imshow('Sobel', sobel.astype(np.uint8))

#destroy Windows
cv.waitKey(0)
cv.destroyAllWindows()
