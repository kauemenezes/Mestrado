import cv2
import numpy as np
import matplotlib.pyplot as plt

FILE_NAME = 'apple.jpg'

img = cv2.imread(FILE_NAME)

(rows, cols) = img.shape[:2]

M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
rotated_image = cv2.warpAffine(img, M, (cols, rows))

plt.figure(3+1, figsize=(10,8))
plt.subplot(121)
plt.imshow(img, 'gray')
plt.axis('OFF')

plt.subplot(122)
plt.imshow(rotated_image, 'gray')
plt.axis('OFF')

plt.show()
