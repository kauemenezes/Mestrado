import numpy as np
import cv2
import matplotlib.pyplot as plt


def calculate_histogram(image):
    output = np.zeros(256)

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            output[(image[i, j])] += 1

    return output


# create our cumulative sum function
def cumulative_sum(a):
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)


def normalize_cumulative_sum(cs):
    # re-normalize cumsum values to be between 0-255
    # numerator & denomenator
    nj = (cs - cs.min()) * 255
    N = cs.max() - cs.min()

    # re-normalize the cdf
    cs = nj / N

    return cs


if __name__ == '__main__':

    image = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
    flat = image.flatten()

    output = calculate_histogram(image)

    # Histogram equalization process
    cs = cumulative_sum(output)
    cs = normalize_cumulative_sum(cs)
    cs = cs.astype('uint8')

    img_new = cs[flat]

    plt.figure(1)
    plt.imshow(image, 'gray')
    # plt.title('Original image')

    plt.figure(2)
    plt.stem(output)
    # plt.title('Image Histogram')

    plt.figure(3)
    plt.hist(img_new, bins=50)
    # plt.title('Histograma equalized')

    plt.figure(4)
    plt.imshow(np.reshape(img_new, image.shape), 'gray')
    # plt.title('New image')

    plt.show()
