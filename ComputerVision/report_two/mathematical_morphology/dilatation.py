import cv2
import numpy as np
import matplotlib.pyplot as plt


def structuring_element (size, type):
    if type == 1:
        # square
        kernel = np.ones(size, np.uint8)
        return kernel
    if type == 2:
        # star
        kernel = np.ones(size, np.uint8)
        center = np.uint8((size[0]/2))
        for k in range(0,size[0]):
            kernel[center,k] = 0
            kernel[k,center] = 0
        return kernel
    if type == 3:
        # something 01
        kernel = np.ones(size, np.uint8)
        center = np.uint8((size[0]/2))
        kernel[:,0] = 0
        kernel[center,1] = 0
        return kernel


# Reading the input image
img = cv2.imread('original.jpg')

size = np.array([3,3])
# kernel = structuring_element(size, 1)
# kernel = structuring_element(size, 2)
kernel = structuring_element(size, 3)

img_dilation = cv2.dilate(img, kernel, iterations=2)

plt.figure(figsize=(9,9))
plt.subplot(131)
plt.imshow(img)
plt.axis("off")
plt.title("Imagem original")

plt.subplot(132)
plt.imshow(img_dilation)
plt.axis("off")
plt.title("Imagem dilatada")

plt.subplot(133)
plt.imshow(kernel, 'gray', interpolation='nearest')
plt.axis("off")
plt.grid()
plt.title("El. Estruturante")
plt.show()