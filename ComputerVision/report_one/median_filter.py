import numpy as np
import cv2
from convolution import convolution
import matplotlib.pyplot as plt


def mean_kernel(size):
    kernel = np.ones((size, size))

    return kernel


def mean_filter(image, kernel_size, iterations, verbose=False):
    kernel = mean_kernel(kernel_size)
    output = np.zeros(image.shape)

    for k in range(0, iterations):
        if k == 0:
            output = convolution(image, kernel, verbose=verbose)
        else:
            image = output
            output = convolution(image, kernel, verbose=verbose)

    return output


def convolution(image, kernel, verbose=False):
    if len(image.shape) == 3:
        print("Found 3 Channels : {}".format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Converted to Gray Channel. Size : {}".format(image.shape))
    else:
        print("Image Shape : {}".format(image.shape))

    print("Kernel Shape : {}".format(kernel.shape))

    # if verbose:
    #     plt.imshow(image, cmap='gray')
    #     plt.title("Image")
    #     plt.figure(1)

    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    # if verbose:
    #     plt.imshow(padded_image, cmap='gray')
    #     plt.title("Padded Image")
    #     plt.figure(2)

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.median(padded_image[row:row + kernel_row, col:col + kernel_col])

    print("Output Image size : {}".format(output.shape))

    if verbose:
        plt.figure(3)
        plt.imshow(output, cmap='gray')
        # plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))

    return output


if __name__ == '__main__':

    image = img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

    mean_filter(image, 9, 1, verbose=True)

    plt.show()

