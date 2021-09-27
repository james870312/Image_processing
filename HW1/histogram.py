import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

#choose read which image
img = cv.imread('Lena.bmp', cv.IMREAD_GRAYSCALE)
#img = cv.imread('Peppers.bmp', cv.IMREAD_GRAYSCALE)
#img = cv.imread('Cameraman.bmp', cv.IMREAD_GRAYSCALE)

#print height and width of input image
height, width = img.shape
print(type(img))
print(img.shape)
print("height = ", height)
print("width = ", width)

#creat new image for output and array of cdf & pdf
h_eq = np.zeros( (height, width), np.uint8 )
pdf = np.zeros( (256))
cdf = np.zeros( (256))

#calculation pdf
print("wait for produce pdf ......")
for k in range(256):
    freq = 0
    for i in range(height):
        for j in range(width):
            if img[i][j] == k: freq += 1
    pdf[k] = freq
print("pdf done")

#calculation cdf
for i in range(256):
    for j in range(i+1):
        cdf[i] = cdf[i] + pdf[j]        
print("cdf done")
print("min of cdf = ", np.min(cdf))

#histogram equalization
for i in range(height):
    for j in range(width):
        h_eq[i][j] = (cdf[img[i][j]] - np.min(cdf))/((height*width) - np.min(cdf))*(256 - 1)
        
print("histogram equalization done")
cv.imshow('input', img)
cv.imshow('histogram_equalization', h_eq)

#show histogram
plt.subplot(2, 1, 1)
plt.hist(img.ravel(), 256, [0, 255],label= 'original image')
plt.subplot(2, 1, 2)
plt.hist(h_eq.ravel(), 256, [0, 255],label= 'histogram_equalization image')
plt.savefig('histogram.png')
plt.show()

#destroy Windows
cv.waitKey(0)
cv.destroyAllWindows()

