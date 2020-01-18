import cv2
import numpy as np
import matplotlib.pyplot as plt

FILE_NAME = 'car.jpg'

M = np.float32([[1, 0, 100], [0, 1, 50]])

img = cv2.imread(FILE_NAME)
(rows, cols) = img.shape[:2]

translated_image = cv2.warpAffine(img, M, (cols, rows))

plt.figure(3+1, figsize=(10,8))
plt.subplot(121)
plt.imshow(img, 'gray')
plt.axis('OFF')

plt.subplot(122)
plt.imshow(translated_image, 'gray')
plt.axis('OFF')

plt.show()
