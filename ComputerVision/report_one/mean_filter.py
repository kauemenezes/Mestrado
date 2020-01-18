import numpy as np
import cv2
from convolution import convolution


def mean_kernel(size):
    kernel = np.ones((size, size))
    kernel = kernel / kernel.sum()

    return kernel


def mean_filter(image, kernel_size, iterations,  verbose=False):
    kernel = mean_kernel(kernel_size)
    output = np.zeros(image.shape)

    for k in range(0, iterations):
        if k == 0:
            output, plt = convolution(image, kernel, average=True, verbose=verbose)
        else:
            image = output
            output, plt = convolution(image, kernel, average=True, verbose=verbose)

    return output, plt


if __name__ == '__main__':

    image = img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

    _, plt = mean_filter(image, 9, 1, verbose=True)
    plt.show()
