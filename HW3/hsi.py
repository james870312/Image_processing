import cv2 as cv
import numpy as np
import math

#choose read which image
#img = cv.imread('aloe.jpg', cv.IMREAD_COLOR)
#img = cv.imread('church.jpg', cv.IMREAD_COLOR)
#img = cv.imread('kitchen.jpg', cv.IMREAD_COLOR)
img = cv.imread('house.jpg', cv.IMREAD_COLOR)

#print height and width of input image
height, width, channel = img.shape
print(type(img.shape))
print(img.shape)
print("height = ", height)
print("width = ", width)
print("channel = ", channel)
print("*************")

#set parameter
normal_img = np.zeros( (height, width, channel), np.float )
buffer = np.zeros( 3, np.float )
hsi_img = np.zeros( (height, width, channel), np.float )
laphsi_img = np.zeros( (height, width, channel), np.float )
hsv_img = np.zeros( (height, width, channel), np.uint8 )
rgb_img = np.zeros( (height, width, channel), np.float )

l_kernel = np.array([[ 0.0,-1.0, 0.0],
                     [-1.0, 5.0,-1.0],
                     [ 0.0,-1.0, 0.0]])

#normalization
normal_img[:,:,:] = img[:,:,:]/255

#RGB transform to HSI space
#H
for i in range(height):
    for j in range(width):
        sqrt = math.sqrt((normal_img[i,j,2] - normal_img[i,j,1])**2
                                +(normal_img[i,j,2] - normal_img[i,j,0])
                                *(normal_img[i,j,1] - normal_img[i,j,0]))
        if(sqrt != 0):
            theta = np.arccos(0.5*(normal_img[i,j,2] - normal_img[i,j,1]+ normal_img[i,j,2] - normal_img[i,j,0])/sqrt)
            if(normal_img[i, j, 0] <= normal_img[i, j, 1]):
                hsi_img[i,j,0] = theta
            elif(normal_img[i, j, 0] > normal_img[i, j, 1]):
                hsi_img[i,j,0] = 2*np.pi - theta
        else: hsi_img[i,j,0] = 0
        hsi_img[i,j,0] = hsi_img[i,j,0]/(2*np.pi)
        
#S
        buffer[:] = normal_img[i, j, :]
        if((normal_img[i, j, 0] + normal_img[i, j, 1] + normal_img[i, j, 2]) == 0):
            hsi_img[i, j, 1] = 0
        else:
            hsi_img[i, j, 1] = 1 - (buffer.min() * 3 / (normal_img[i, j, 0] + normal_img[i, j, 1] + normal_img[i, j, 2]) )

#I
hsi_img[:,:,2] = (normal_img[:,:,0] + normal_img[:,:,1] + normal_img[:,:,2])/3

#non-ormalization
hsi_img[:,:,:] = hsi_img[:,:,:]*255
hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

#run convolution by laplacian kernel
print("wait for convolution ......")


laphsi_img[:, :, :] = hsi_img[:, :, :]
'''
for i in range(height-2):
    for j in range(width-2):
        laphsi_img[i+1, j+1, 2] = np.sum(l_kernel * hsi_img[i:i+3, j:j+3, 2])
        if(laphsi_img[i+1, j+1, 2] < 0):laphsi_img[i+1, j+1, 2] = 0
        if(laphsi_img[i+1, j+1, 2] >= 255):laphsi_img[i+1, j+1, 2] = 255
'''

for i in range(height-2):
    for j in range(width-2):
        for k in range(channel-1):
            laphsi_img[i+1, j+1, k+1] = np.sum(l_kernel * hsi_img[i:i+3, j:j+3, k+1])
            if(laphsi_img[i+1, j+1, k+1] < 0):laphsi_img[i+1, j+1, k+1] = 0
            if(laphsi_img[i+1, j+1, k+1] >= 255):laphsi_img[i+1, j+1, k+1] = 255




print("convolution done")

#normalization
normal_img[:,:,:] = laphsi_img[:,:,:]/255

#HSI transform to RGB space
for i in range(height):
    for j in range(width):
        #if((normal_img[i,j,0]*180/math.pi)>=0 and (normal_img[i,j,0]*180/math.pi)<120):
        if(normal_img[i,j,0]*2*math.pi >= 0 and normal_img[i,j,0]*2*math.pi < 2*math.pi/3):
            #B
            rgb_img[i,j,0] = normal_img[i,j,2] * (1 - normal_img[i,j,1])
            #R
            rgb_img[i,j,2] = normal_img[i,j,2] * (1 + (normal_img[i,j,1]*np.cos(normal_img[i,j,0]*2*math.pi)/np.cos(math.pi/3 - normal_img[i,j,0]*2*math.pi)))      
            #G
            rgb_img[i,j,1] = 3 * normal_img[i,j,2] - (rgb_img[i,j,2] + rgb_img[i,j,0])

        #elif((hsi_img[i,j,0]*180/math.pi)>=120 and (hsi_img[i,j,0]*180/math.pi)<240):
        if(normal_img[i,j,0]*2*math.pi >= 2*math.pi/3 and normal_img[i,j,0]*2*math.pi < 4*math.pi/3):
            #R
            rgb_img[i,j,2] = normal_img[i,j,2] * (1 - normal_img[i,j,1])
            #G
            rgb_img[i,j,1] = normal_img[i,j,2] * (1 + (normal_img[i,j,1]*math.cos(normal_img[i,j,0]*2*math.pi-2*math.pi/3)/math.cos(math.pi - normal_img[i,j,0]*2*math.pi)))
            #B
            rgb_img[i,j,0] = 3 * normal_img[i,j,2] - (rgb_img[i,j,2] + rgb_img[i,j,1])
            
        #elif((hsi_img[i,j,0]*180/math.pi)>=240 and (hsi_img[i,j,0]*180/math.pi)<360):
        if(normal_img[i,j,0]*2*math.pi >= 4*math.pi/3 and normal_img[i,j,0]*2*math.pi < 6*math.pi/3):
            #G
            rgb_img[i,j,1] = normal_img[i,j,2] * (1 - normal_img[i,j,1])
            #B
            rgb_img[i,j,0] = normal_img[i,j,2] * (1 + (normal_img[i,j,1]*math.cos(normal_img[i,j,0]*2*math.pi-4*math.pi/3)/math.cos(5*math.pi/3 - normal_img[i,j,0]*2*math.pi)))
            #R
            rgb_img[i,j,2] = 3 * normal_img[i,j,2] - (rgb_img[i,j,1] + rgb_img[i,j,0])

#non-ormalization
rgb_img[:,:,:] = rgb_img[:,:,:]*255

#show image
cv.imshow('source', img)
#cv.imshow('normalized', normal_img)
cv.imshow('HSI', hsi_img.astype(np.uint8))
cv.imshow('LAP_HSI', laphsi_img.astype(np.uint8))
cv.imshow('LIB_HSV', hsv_img)
cv.imshow('RGB', rgb_img.astype(np.uint8))
#cv.imshow('RGB2', rgb_img)

#destroy Windows
cv.waitKey(0)
cv.destroyAllWindows()
