import cv2
import numpy as np
import matplotlib.pyplot as plt

FILE_NAME = 'apple.jpg'

img = cv2.imread(FILE_NAME)

(height, width) = img.shape[:2]

resized_image = cv2.resize(img, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)

plt.figure(3+1, figsize=(10,8))
plt.subplot(121)
plt.imshow(img, 'gray')
plt.title('Imagem original %dx%d' % (img.shape[0], img.shape[1]))
plt.axis('OFF')

plt.subplot(122)
plt.imshow(resized_image, 'gray')
plt.title('Imagem reduzida %dx%d' % (resized_image.shape[0], resized_image.shape[1]))
plt.axis('OFF')

plt.show()

