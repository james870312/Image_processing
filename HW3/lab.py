import cv2 as cv
import numpy as np
import math

#choose read which image
#img = cv.imread('aloe.jpg', cv.IMREAD_COLOR)
#img = cv.imread('church.jpg', cv.IMREAD_COLOR)
#img = cv.imread('kitchen.jpg', cv.IMREAD_COLOR)
img = cv.imread('house.jpg', cv.IMREAD_COLOR)

#set parameter
rgb2xyz = np.array([[0.412453, 0.357580, 0.180423],
                    [0.212671, 0.715160, 0.072169],
                    [0.019334, 0.119193, 0.950227]])

xyz2rgb = np.array([[ 3.240479,-1.537150,-0.498535],
                    [-0.969256, 1.875992, 0.041556],
                    [ 0.055648,-0.204043, 1.057311]])

l_kernel = np.array([[ 0.0,-1.0, 0.0],
                     [-1.0, 5.0,-1.0],
                     [ 0.0,-1.0, 0.0]])

xyz_n = np.array([0.9515, 1.0000, 1.0886])
buffer = np.zeros( 3, np.float )
f_xyz = np.zeros( 3, np.float )

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
xyz_img = np.zeros( (height, width, channel), np.float )
laplab_img = np.zeros( (height, width, channel), np.float )
lab_img = np.zeros( (height, width, channel), np.float )
liblab_img = np.zeros( (height, width, channel), np.uint8 )
rgb_img = np.zeros( (height, width, channel), np.float )

#normalization
normal_img[:,:,:] = img[:,:,:]/255

#RGB transform to XYZ space
for i in range(height):
    for j in range(width):
        xyz_img[i, j, :] = np.dot(rgb2xyz, normal_img[i,j,:])

#XYZ transform to L*A*B space
for i in range(height):
    for j in range(width):
        buffer[:] = xyz_img[i, j, :]/xyz_n[:]
        
        for k in range(3):
            if(buffer[k] > 0.008856):
                f_xyz[k] = math.pow(buffer[k], 1/3)
            else:
                f_xyz[k] = 7.787 * buffer[k] +16/116
        #L*
        if(buffer[1] > 0.008856):
            lab_img[i, j, 0] = 116 * math.pow(buffer[1], 1/3) - 16
        else:
            lab_img[i, j, 0] = 903.3 * buffer[1]
        
        #A*
        lab_img[i, j, 1] = 500 * (f_xyz[0] - f_xyz[1])
        
        #B*
        lab_img[i, j, 2] = 200 * (f_xyz[1] - f_xyz[2])
        
liblab_img = cv.cvtColor(img, cv.COLOR_BGR2LAB)

#run convolution by laplacian kernel
print("wait for convolution ......")
laplab_img[:, :, :] = lab_img[:, :, :]

for i in range(height-2):
    for j in range(width-2):
        laplab_img[i+1, j+1, 0] = np.sum(l_kernel * lab_img[i:i+3, j:j+3, 0])
        if(laplab_img[i+1, j+1, 0] < 0):laplab_img[i+1, j+1, 0] = 0
        if(laplab_img[i+1, j+1, 0] >= 255):laplab_img[i+1, j+1, 0] = 255

'''
for i in range(height-2):
    for j in range(width-2):
        for k in range(channel):
            laplab_img[i+1, j+1, k] = np.sum(l_kernel * lab_img[i:i+3, j:j+3, k])
            if(laplab_img[i+1, j+1, k] < 0):laplab_img[i+1, j+1, k] = 0
            if(laplab_img[i+1, j+1, k] >= 255):laplab_img[i+1, j+1, k] = 255
'''
print("convolution done")

#L*A*B transform to XYZ space
for i in range(height):
    for j in range(width):
        f_xyz[1] = (laplab_img[i, j, 0] + 16) / 116
        f_xyz[0] =  f_xyz[1] + laplab_img[i, j, 1]/500
        f_xyz[2] =  f_xyz[1] - laplab_img[i, j, 2]/200
        #X
        if(f_xyz[0] > 0.008856):
            xyz_img[i, j, 0] = xyz_n[0] * (f_xyz[0]**3)
        else:
            xyz_img[i, j, 0] = ((f_xyz[0]-16) / 116) * 3 * (0.008865**2) * xyz_n[0]
        #Y
        if(f_xyz[1] > 0.008856):
            xyz_img[i, j, 1] = xyz_n[1] * (f_xyz[1]**3)
        else:
            xyz_img[i, j, 1] = ((f_xyz[1]-16) / 116) * 3 * (0.008865**2) * xyz_n[1]
        #Z
        if(f_xyz[2] > 0.008856):
            xyz_img[i, j, 2] = xyz_n[2] * (f_xyz[2]**3)
        else:
            xyz_img[i, j, 2] = ((f_xyz[2]-16) / 116) * 3 * (0.008865**2) * xyz_n[2]
      
#XYZ transform to RGB space
for i in range(height):
    for j in range(width):
        rgb_img[i, j, :] = np.dot(xyz2rgb, xyz_img[i,j,:])

#non-ormalization
rgb_img[:,:,:] = rgb_img[:,:,:] * 255

#show image
cv.imshow('source', img)
#cv.imshow('normalized', normal_img)
#cv.imshow('CIE_XYZ', xyz_img)
cv.imshow('LAB', lab_img.astype(np.uint8))
cv.imshow('LAP_LAB', laplab_img.astype(np.uint8))
cv.imshow('lib_LAB', liblab_img)
cv.imshow('RGB', rgb_img.astype(np.uint8))
#cv.imshow('RGB2', rgb_img)

#destroy Windows
cv.waitKey(0)
cv.destroyAllWindows()
