import numpy as np
import cv2
import matplotlib.pyplot as plt


def laplacian_kernel(size):
    kernel = np.ones((size, size))
    kernel[int(size/2), int(size/2)] = -1*(np.sum(kernel)-1)

    return kernel


def laplacian_filter(image, kernel_size, iterations, verbose=False):
    kernel = laplacian_kernel(kernel_size)
    output = np.zeros(image.shape)

    for k in range(0, iterations):
        if k == 0:
            output, plt = convolution(image, kernel, average=True, verbose=verbose)
        else:
            image = output
            output, plt = convolution(image, kernel, average=True, verbose=verbose)

    return output, plt


def convolution(image, kernel, average=True, verbose=False):
    if len(image.shape) == 3:
        print("Found 3 Channels : {}".format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Converted to Gray Channel. Size : {}".format(image.shape))
    else:
        print("Image Shape : {}".format(image.shape))

    print("Kernel Shape : {}".format(kernel.shape))

    if verbose:
        plt.figure(1)
        plt.imshow(image, cmap='gray')
        plt.title("Image")

    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

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
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]

    print("Output Image size : {}".format(output.shape))

    output = np.uint8(output)

    if verbose:
        plt.figure(3)
        plt.imshow(output, cmap='gray')
        # plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))

    return output, plt


if __name__ == '__main__':

    image = img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

    _, plt = laplacian_filter(image, 9, 1, verbose=True)
    plt.show()
