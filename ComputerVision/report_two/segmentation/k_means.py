import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_result(image, result, title):
    fig = plt.figure(figsize=(12, 12), dpi=80)
    a = fig.add_subplot(1, 2, 1)
    a.axis('off')
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    a.set_title('Original')

    a = fig.add_subplot(1, 2, 2)
    a.axis('off')
    plt.imshow(result, cmap=plt.get_cmap('gray'))
    a.set_title(title)

    plt.show()


def kmeans(image):
    img = image
    Z = img.reshape((-1, 2))
    Z = np.float32(Z)

    K = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    aux = center[label.flatten()]
    result = aux.reshape((img.shape))

    plot_result(result, 'K-means (K={})'.format(K))


FILE_NAME = 'lena.bmp'

img = cv2.imread(FILE_NAME)
kmeans(img)
