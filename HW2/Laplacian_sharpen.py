import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.fftpack import fft,ifft

#choose read which image
img = cv.imread('blurry_moon.tif', cv.IMREAD_GRAYSCALE)
#img = cv.imread('skeleton_orig.bmp', cv.IMREAD_GRAYSCALE)

#print height and width of input image
height, width = img.shape
print(type(img))
print(img.shape)
print("height = ", height)
print("width = ", width)

#Fourier transform
fft2 = np.fft.fft2(img)
lap_fft2 = np.fft.fft2(img)
plt.subplot(231)
plt.imshow(np.abs(fft2),'gray'),plt.title('fft2')

#Centralization
shift2center = np.fft.fftshift(fft2)
shift2center[int((height/2)-1) : int((height/2)+1), int((width/2)-1) : int((width/2)+1)] = 0
plt.subplot(232)
plt.imshow(np.abs(shift2center),'gray'),plt.title('shift2center')

#laplacian sharpening
for i in range(height):
    for j in range(width):
        lap_fft2[i][j] = -4*(math.pi**2)*abs((i-height/2)**2 + (j-width/2)**2)*shift2center[i][j]
plt.subplot(233)
plt.imshow(np.abs(lap_fft2),'gray'),plt.title('lap_fft2')

#Inverse Centralization
center2shift = np.fft.ifftshift(lap_fft2)
plt.subplot(234)
plt.imshow(np.abs(center2shift),'gray'),plt.title('center2shift')

#Inverse Fourier transform
ifft2 = np.fft.ifft2(center2shift)
plt.subplot(235)
plt.imshow(np.abs(ifft2),'gray'),plt.title('ifft2')

#normalization & image enhancement
lap_img = np.abs(ifft2)/np.max(np.abs(ifft2))
result_img = lap_img + (img/255)

#show image
plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)
cv.imshow('input image', img)
cv.imshow('laplacian', lap_img)
cv.imshow('image enhancement by laplacian', result_img)

#destroy Windows
cv.waitKey(0)
cv.destroyAllWindows()
