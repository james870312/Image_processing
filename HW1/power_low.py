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

#creat new image for output
p_law = np.zeros( (height, width), np.uint8 )

#power-law (gamma) transformation
# s = cr^y
#recommend y = 0.4 for Lena.bmp
#recommend y = 6   for Peppers.bmp
#recommend y = 1.2 for Cameraman.bmp
c = 255
y = 0.4

for i in range(height):
    for j in range(width):
        p_law[i][j] = c*((img[i][j]/c)**y)

cv.imshow('input', img)
cv.imshow('power_law', p_law)

#show histogram
plt.subplot(2, 1, 1)
plt.hist(img.ravel(), 256, [0, 255],label= 'original image')
plt.subplot(2, 1, 2)
plt.hist(p_law.ravel(), 256, [0, 255],label= 'power_law image')

#destroy Windows
cv.waitKey(0)
cv.destroyAllWindows()
