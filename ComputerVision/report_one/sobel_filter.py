import numpy as np
import cv2
import matplotlib.pyplot as plt
import math as m


def sobel_kernel(size):
    kernel_horizontal = np.zeros((size, size))
    kernel_horizontal[:, 0] = -1
    kernel_horizontal[:, (size - 1)] = 1

    kernel_vertical = np.zeros((size, size))
    kernel_vertical[0, :] = -1
    kernel_vertical[(size - 1), :] = 1

    kernel_horizontal = np.zeros((size, size))
    kernel_horizontal[:, 0] = -1
    kernel_horizontal[int(size / 2), 0] = -2
    kernel_horizontal[:, (size - 1)] = 1
    kernel_horizontal[int(size / 2), (size - 1)] = 2

    kernel_vertical = np.zeros((size, size))
    kernel_vertical[0, :] = -1
    kernel_vertical[0, int(size / 2)] = -2
    kernel_vertical[(size - 1), :] = 1
    kernel_vertical[(size - 1), int(size / 2)] = 2

    return kernel_horizontal, kernel_vertical


def sobel_filter(image, kernel_size, verbose=False):
    kernel_horizontal, kernel_vertical = sobel_kernel(kernel_size)
    return convolution(image, kernel_horizontal, kernel_vertical, verbose=verbose)


def convolution(image, kernel_horizontal, kernel_vertical, verbose=False):
    if len(image.shape) == 3:
        print("Found 3 Channels : {}".format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Converted to Gray Channel. Size : {}".format(image.shape))
    else:
        print("Image Shape : {}".format(image.shape))

    print("Kernel Shape : {}".format(kernel_horizontal.shape))

    if verbose:
        plt.figure(1)
        plt.imshow(image, cmap='gray')
        plt.title("Image")

    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel_horizontal.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    if verbose:
        plt.figure(2)
        plt.imshow(padded_image, cmap='gray')
        plt.title("Padded Image")

    for row in range(image_row):
        for col in range(image_col):
            horizontal_sum = np.sum(kernel_horizontal * padded_image[row:row + kernel_row, col:col + kernel_col])
            vertical_sum = np.sum(kernel_vertical * padded_image[row:row + kernel_row, col:col + kernel_col])
            horizontal = m.ceil((horizontal_sum / (kernel_horizontal.shape[0] * kernel_horizontal.shape[1])))
            vertical = m.ceil((vertical_sum / (kernel_vertical.shape[0] * kernel_vertical.shape[1])))
            output[row, col] = np.sqrt(horizontal**2 + vertical**2)

    print("Output Image size : {}".format(output.shape))

    output = np.uint8(output)

    if verbose:
        plt.figure(3)
        plt.imshow(output, cmap='gray')
        # plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))

    return output, plt


if __name__ == '__main__':

    image = img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

    _, plt = sobel_filter(image, 9, verbose=True)
    plt.show()
