import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


def getSeed(w, h):
    p = 10.

    pos_ini_x_mrk = int(w / 2 - p * w / 100.)
    pos_ini_y_mrk = int(h / 2 - p * h / 100.)
    pos_fim_x_mrk = int(w / 2 + p * w / 100.)
    pos_fim_y_mrk = int(h / 2 + p * h / 100.)

    seed = np.zeros(shape=(w, h), dtype=np.uint8)
    seed[pos_ini_x_mrk:pos_fim_x_mrk, pos_ini_y_mrk:pos_fim_y_mrk] = 255

    return seed


def neighbors(x, y, w, h):
    points = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1),
              (x - 1, y + 1), (x + 1, y + 1), (x - 1, y - 1), (x + 1, y - 1)]

    list_ = deque()
    for p in points:
        if (p[0] >= 0 and p[1] >= 0 and p[0] < w and p[1] < h):
            list_.append(p)

    return list_


def region_growth(image, epsilon=10):
    image = cv2.blur(image, (5, 5))
    w, h = image.shape

    reg = getSeed(w, h)
    queue = deque()
    for x in range(w):
        for y in range(h):
            if reg[x, y] == 255:
                queue.append((x, y))

    while queue:
        point = queue.popleft()
        x = point[0]
        y = point[1]

        v_list = neighbors(x, y, w, h)
        for v in v_list:
            v_x = v[0]
            v_y = v[1]
            if ((reg[v_x][v_y] != 255) and (abs(int(image[x][y]) - int(image[v_x][v_y])) < epsilon)):
                reg[v_x][v_y] = 255
                queue.append(v)

    plot_result(reg, 'Crescimento de Região')


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


FILE_NAME = 'lena.bmp'
# FILE_NAME = 'car.jpg'

img = cv2.imread(FILE_NAME)
region_growth(img)
