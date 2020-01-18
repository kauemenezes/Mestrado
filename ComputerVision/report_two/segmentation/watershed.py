import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.morphology import disk, square, star


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


def getMarkers(m, n):
    markers = np.zeros([m, n])
    # Center
    m = int(m / 2)
    n = int(n / 2)
    markers[20:40, 20:40] = 200
    markers[m:m + 20, n:n + 20] = 100

    return markers


def watershed(image):
    image = image
    image_ext = morphology.dilation(image, disk(5)) - image

    m, n = image.shape
    markers = getMarkers(m, n)
    ws = morphology.watershed(image_ext, markers)

    plotWatershed(image, 255 - image_ext, markers, ws)


def plotWatershed(image, dilation, markers, watershed):
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=4, figsize=(8, 2.5), sharex=True, sharey=True)

    ax0.imshow(image, cmap=plt.get_cmap('gray'))
    ax0.set_title('Original')
    ax0.axis('off')

    ax1.imshow(dilation, cmap=plt.get_cmap('gray'))
    ax1.set_title('External Dilation')
    ax1.axis('off')

    ax2.imshow(markers, cmap=plt.get_cmap('gray'))
    ax2.set_title('Markers')
    ax2.axis('off')

    ax3.imshow(watershed, cmap=plt.get_cmap('nipy_spectral'), interpolation='nearest')
    ax3.set_title('Watershed')
    ax3.axis('off')

    fig.tight_layout()
    plt.show()


FILE_NAME = 'lena.bmp'

img = cv2.imread(FILE_NAME)
watershed(img)
