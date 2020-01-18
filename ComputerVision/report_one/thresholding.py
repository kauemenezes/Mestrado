import numpy as np
import cv2
import matplotlib.pyplot as plt


def thresholding(image, threshold):
    image_row, image_col = image.shape
    output = np.zeros(image.shape)
    threshold_min = np.min(image)
    threshold_max = np.max(image)

    for row in range(image_row):
        for col in range(image_col):
            if image[row, col] > threshold:
                output[row, col] = threshold_max
            else:
                output[row, col] = threshold_min

    return output


def get_thresholded_image(image, iterations):
    output = np.zeros(image.shape)

    for k in range(0, iterations):
        if k == 0:
            output = thresholding(image, 110)
        else:
            image = output
            output = thresholding(image, 110)

    return output


if __name__ == '__main__':

    image = img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

    output = get_thresholded_image(image, 1)

    plt.figure(1)
    plt.imshow(output, cmap='gray')
    # plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))
    plt.show()
