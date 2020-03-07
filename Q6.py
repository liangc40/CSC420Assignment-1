import matplotlib.pyplot as plt
import random
import Q4
import cv2
import numpy as np

np.seterr(over='ignore')


def addRandomNoise(img, m):
    """
    Add random noise to an image.
    :param img: An image represented by matrix
    :type img: [[int]]
    :param m: magnitude of noise
    :type m: float
    :return: an image with random noise on it
    :rtype: [[int]]
    """
    length, height = img.shape
    for i in range(length):
        for j in range(height):
            potential_value = img[i][j] * (1 + random.uniform(-m, m))
            # if the potential value is larger than 255 or 0, set it to boundary
            if potential_value > 255:
                img[i][j] = 255
            elif potential_value < 0:
                img[i][j] = 0
            else:
                img[i][j] = potential_value
    return img


def removeNoise(img, kernel):
    """
    Use the given kernel to reduce the level of noise on an image
    :param img: the input image represented by matrix
    :type img: [[int]]
    :param kernel: the kernel represented by matrix
    :type kernel: [[int]]
    :return: the noise reduced image
    :rtype: [[int]]
    """
    processed_image = Q4.MyCorrelation(img, kernel, 'same')
    return processed_image


def addSaltAndPepperNoise(img, density):
    """
    Add salt and pepper noise to an image.
    :param img: the input image represented by matrix
    :type img: [[int]]
    :param density: the density of salt and pepper noise
    :type density: float
    :return: an image with certain density of salt and pepper noise on it
    :rtype: [[int]]
    """
    output = np.zeros(img.shape, np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            random_num = random.random()
            if random_num < density / 2:
                output[i][j] = 0
            elif random_num > 1 - density / 2:
                output[i][j] = 255
            else:
                output[i][j] = img[i][j]
    return output


def add_Surrounding_Pixels(img, length, height, length_bound, height_bound, size):
    """
    Find the surrounding pixels of a certain pixel.
    :param size: an odd number indicating the size of my filter
    :type size: int
    :param img: the input image
    :type img: [[int]]
    :param length: an integer indicating the length of the upper left corner of surrounding pixels
    :type length: int
    :param height: an integer indicating the height of the upper left corner of surrounding pixels
    :type height: int
    :param length_bound: an integer indicating the length of the input image
    :type length_bound: int
    :param height_bound: an integer indicating the height of the input image
    :type height_bound: int
    :return: an array of surrounding pixels
    :rtype: [int]
    """
    surrounding_pixels = []
    current_length = length
    while current_length < length + size:
        current_height = height
        while current_height < height + size:
            if 0 <= current_length < length_bound and 0 <= current_height < height_bound:
                surrounding_pixels.append(img[int(current_length)][int(current_height)])
            current_height += 1
        current_length += 1
    return surrounding_pixels


def not_consec_pure_colour(img, i, j, length_bound, height_bound):
    """
    Tell whether the consecutive pixels has similar colours as the input pixel.
    :param img: the input image
    :type img: [[int]]
    :param i: an integer indicating the row of the given pixel
    :type i: int
    :param j: an integer indicating the column of the given pixel
    :type j: int
    :param length_bound: an integer indicating the length of the input image
    :type length_bound:  int
    :param height_bound: an integer indicating the height of the input image
    :type height_bound: int
    :return: a boolean telling whether the consecutive pixels has similar
             colours as the input pixel
    :rtype: boolean
    """
    result = True
    if i - 1 >= length_bound:
        if abs(img[i - 1][j] - img[i][j]) <= 3:
            result = False
    if i + 1 < height_bound:
        if abs(img[i + 1][j] - img[i][j]) <= 3:
            result = False
    if j - 1 >= height_bound:
        if abs(img[i][j - 1] - img[i][j]) <= 3:
            result = False
    if j + 1 < height_bound:
        if abs(img[i][j + 1] - img[i][j]) <= 3:
            result = False
    return result


def myMedianFilter(img, size):
    """
    create a median filter especially remove the salt and pepper noise
    :param img: the input image
    :type img: [[int]]
    :param size: the size of the filter
    :type size: int
    :return: a filtered image
    :rtype: [[int]]
    """
    length, height = img.shape
    new_img = np.zeros((length, height))
    for i in range(length):
        for j in range(height):
            if 0 < img[i][j] < 255 and not_consec_pure_colour(img, i, j, length, height):
                new_img[i][j] = img[i][j]
            else:
                surrounding_pixels = add_Surrounding_Pixels(img, int(i - (size - 1) / 2), int(j - (size - 1) / 2), length, height, size)
                new_img[i][j] = np.median(surrounding_pixels)
    return new_img


if __name__ == '__main__':
    # (a) add random noise to the image
    img = cv2.imread('gray.jpg', 0)
    img2 = addRandomNoise(img, 0.05)
    plt.imshow(img2, cmap='gray')
    plt.show()

    # (b) the filter I chose is the mean filter
    # create a mean filter
    kernel = np.ones((3, 3), np.float32) / 9
    img3 = cv2.filter2D(img2, -1, kernel)
    plt.imshow(img3, cmap='gray')
    plt.show()

    # (c) add salt and pepper noise to the image
    img4 = addSaltAndPepperNoise(img, 0.05)
    plt.imshow(img4, cmap='gray')
    plt.show()

    # (d) use median filer to remove salt and pepper noise
    img5 = cv2.medianBlur(img4, 5)
    plt.imshow(img5, cmap='gray')
    plt.show()

    # (e) use myMedianFilter to filter out salt and pepper noise
    img6 = cv2.imread('color.jpg')
    img6 = cv2.cvtColor(img6, cv2.COLOR_BGR2RGB)
    img6 = addSaltAndPepperNoise(img6, 0.05)
    img6[:, :, 0] = myMedianFilter(img6[:, :, 0], 3)
    img6[:, :, 1] = myMedianFilter(img6[:, :, 1], 3)
    img6[:, :, 2] = myMedianFilter(img6[:, :, 2], 3)
    plt.imshow(img6)
    plt.show()
