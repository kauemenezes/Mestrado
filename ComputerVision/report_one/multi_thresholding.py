import numpy as np
import cv2
import matplotlib.pyplot as plt


def multi_thresholding(image, multi_threshold, multi_range):
    image_row, image_col = image.shape
    output = np.zeros(image.shape)

    if multi_threshold.size == 2:
        threshold_2 = multi_threshold[1]
        threshold_1 = multi_threshold[0]
        value_min = 0
        value_med = multi_range[0]
        value_max = 255

    if multi_threshold.size == 3:
        threshold_3 = multi_threshold[2]
        threshold_2 = multi_threshold[1]
        threshold_1 = multi_threshold[0]
        value_min = 0
        value_med_1 = multi_range[0]
        value_med_2 = multi_range[1]
        value_max = 255

    for row in range(image_row):
        for col in range(image_col):
            if multi_threshold.size == 3:
                if image[row, col] > threshold_3:
                    output[row, col] = value_max
                elif image[row, col] <= threshold_3 and image[row, col] > threshold_2:
                    output[row, col] = value_med_1
                elif image[row, col] <= threshold_2 and image[row, col] > threshold_1:
                    output[row, col] = value_med_2
                elif image[row, col] <= threshold_1:
                    output[row, col] = value_min

            if multi_threshold.size == 2:
                if image[row, col] > threshold_2:
                    output[row, col] = value_max
                elif image[row, col] <= threshold_2 and image[row, col] > threshold_1:
                    output[row, col] = value_med
                elif image[row, col] <= threshold_1:
                    output[row, col] = value_min

    return output


def get_thresholded_image(image, iterations, multi_threshold, multi_range):
    output = np.zeros(image.shape)

    for k in range(0, iterations):
        if k == 0:
            output = multi_thresholding(image, multi_threshold, multi_range)
        else:
            image = output
            output = multi_thresholding(image, multi_threshold, multi_range)

    return output


if __name__ == '__main__':

    image = img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

    multi_threshold = np.array([80, 120])
    multi_range = np.array([127])

    output = get_thresholded_image(image, 1, multi_threshold, multi_range)

    plt.figure(1)
    plt.imshow(output, cmap='gray')
    # plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))
    plt.show()
