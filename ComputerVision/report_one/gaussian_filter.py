import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
import math
from convolution import convolution


def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)


def gaussian_kernel(size, sigma=1):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

    kernel_2D *= 1.0 / kernel_2D.max()

    # if verbose:
    #     plt.imshow(kernel_2D, interpolation='none', cmap='gray')
    #     plt.title("Kernel ( {}X{} )".format(size, size))
    #     plt.show()

    return kernel_2D


def gaussian_blur(image, kernel_size, iterations, verbose=False):
    kernel = gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size))
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

    _, plt = gaussian_blur(image, 5, 1, verbose=True)
    plt.show()
