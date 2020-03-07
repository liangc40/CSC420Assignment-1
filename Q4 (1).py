import matplotlib.pyplot as plt
import numpy as np
import cv2


def handle_valid(I, h, filter_length, filter_height, image_length, image_height):
    """
    This functions takes in a grayscale image I in matrix representation, a filter h which is also
    in matrix representation, the length and height of image I, and the length and height of filter h,
    and the function returns the result of applying filter h on image I, and such image has a valid border.
    :param I: an image I represented by matrix
    :type I: [[str]]
    :param h: a filter h represented by matrix
    :type h: [[str]]
    :param filter_length: the length of image I
    :type filter_length: int
    :param filter_height: the height of image I
    :type filter_height: int
    :param image_length: the length of filter h
    :type image_length: int
    :param image_height: the height of filter h
    :type image_height: int
    :return: the returned image represented by matrix
    :rtype: [[int]]
    """
    # find the location of the first pixel where we can apply filter on
    starting_length = (filter_length - 1) / 2
    starting_height = (filter_height - 1) / 2
    # find the location of the last pixel where we can apply filter on
    ending_length = image_length - (filter_length - 1) / 2
    ending_height = image_height - (filter_height - 1) / 2
    # loop through every single pixel where we can apply the filter and do the change
    current_length = starting_length
    new_image = np.zeros((image_length, image_height))
    while current_length < ending_length:
        current_height = starting_height
        while current_height < ending_height:
            temp_sum = 0
            for i in range(filter_length):
                for j in range(filter_height):
                    temp_sum += I[int(current_length + i - (filter_length - 1) / 2), int(
                        current_height + j - (filter_height - 1) / 2)] * h[i, j]
            new_image[int(current_length), int(current_height)] = temp_sum
            current_height = current_height + 1
        current_length = current_length + 1
    # return the resulting image
    return new_image[int(starting_length): int(ending_length), int(starting_height): int(ending_height)]


def handle_same(I, h, filter_length, filter_height, image_length, image_height):
    """
    This functions takes in a grayscale image I in matrix representation, a filter h which is also
    in matrix representation, the length and height of image I, and the length and height of filter h,
    and the function returns the result of applying filter h on image I, and such image has the same border.
    :param I: an image I represented by matrix
    :type I: [[int]]
    :param h: a filter h represented by matrix
    :type h: [[int]]
    :param filter_length: the length of image I
    :type filter_length: int
    :param filter_height: the height of image I
    :type filter_height: int
    :param image_length: the length of filter h
    :type image_length: int
    :param image_height: the height of filter h
    :type image_height: int
    :return: the returned image represented by matrix
    :rtype: [[int]]
    """
    # we first extend the graph to a larger graph where we could call handle_valid() on
    new_image_length = image_length + filter_length - 1
    new_image_height = image_height + filter_height - 1
    unchopped_image = np.zeros((new_image_length, new_image_height))
    unchopped_image[int((filter_length - 1) / 2): int((filter_length - 1) / 2 + image_length),
    int((filter_height - 1) / 2): int((filter_height - 1) / 2 + image_height)] = I
    new_chopped_image = handle_valid(unchopped_image, h, filter_length, filter_height, new_image_length,
                                     new_image_height)
    return new_chopped_image


def handle_full(I, h, filter_length, filter_height, image_length, image_height):
    """
    This functions takes in a grayscale image I in matrix representation, a filter h which is also
    in matrix representation, the length and height of image I, and the length and height of filter h,
    and the function returns the result of applying filter h on image I, and such image has the same border.
    :param I: an image I represented by matrix
    :type I: [[int]]
    :param h: a filter h represented by matrix
    :type h: [[int]]
    :param filter_length: the length of image I
    :type filter_length: int
    :param filter_height: the height of image I
    :type filter_height: int
    :param image_length: the length of filter h
    :type image_length: int
    :param image_height: the height of filter h
    :type image_height: int
    :return: the returned image represented by matrix
    :rtype: [[int]]
    """
    # we first extend the graph to a larger graph where we could call handle_valid() on
    new_image_length = image_length + 2 * (filter_length - 1)
    new_image_height = image_height + 2 * (filter_height - 1)
    unchopped_image = np.zeros((new_image_length, new_image_height))
    unchopped_image[int(filter_length - 1): int(filter_length - 1 + image_length),
    int(filter_height - 1): int(filter_height - 1 + image_height)] = I
    new_chopped_image = handle_valid(unchopped_image, h, filter_length, filter_height, new_image_length,
                                     new_image_height)
    return new_chopped_image


def MyCorrelation(I, h, mode):
    """
    This functions takes in a grayscale image I in matrix representation, a filter h which is also
    in matrix representation, and the function returns the result of applying filter h on image I,
    and such image has the same border.
    :param I: an image I represented by matrix
    :type I: [[str]]
    :param h: an image h represented by matrix
    :type h: [[str]]
    :param mode: an string indicating which kind of border the returned graph has
    :type mode: str
    :return: the returned image represented by matrix
    :rtype: [[int]]
    """
    # get the shape of the graph
    image_length = I.shape[0]
    image_height = I.shape[1]
    filter_length = h.shape[0]
    filter_height = h.shape[1]
    # handle three different cases respectively
    if mode == 'valid':
        return handle_valid(I, h, filter_length, filter_height, image_length, image_height)
    elif mode == 'same':
        return handle_same(I, h, filter_length, filter_height, image_length, image_height)
    elif mode == 'full':
        return handle_full(I, h, filter_length, filter_height, image_length, image_height)
    else:
        raise NameError('Invalid Name')


def reverse_convolution(h):
    """
    Reverse the convolution.
    :param h: a convolution represented by matrix
    :type h: [[int]]
    :return: a reversed convolution
    :rtype: [[int]]
    """
    h_length, h_height = h.shape
    reversed_h = np.zeros((h_length, h_height))
    for i in range(h_length):
        for j in range(h_height):
            reversed_h[h_length - i - 1][h_height - j - 1] = h[i][j]
    return reversed_h


def MyConvolution(I, h, mode):
    reversed_h = reverse_convolution(h)
    resulting_img = MyCorrelation(I, reversed_h, mode)
    return resulting_img


if __name__ == '__main__':
    # (a) Test sample in matrix representation
    img = np.array([[90, 0, 90, 0, 90], [90, 0, 90, 0, 90], [90, 0, 90, 0, 90], [90, 0, 90, 0, 90], [90, 0, 90, 0, 90]])
    kernel = np.array([[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]])

    matrix_valid = MyCorrelation(img, kernel, "valid")
    print("Valid:{} ".format(matrix_valid))
    matrix_same = MyCorrelation(img, kernel, "same")
    print("Same:{} ".format(matrix_same))
    matrix_full = MyCorrelation(img, kernel, "full")
    print("Full:{} ".format(matrix_full))

    # (a) Test sample in actual grayscale graph
    img = cv2.imread('mona_lisa.jpg', 0)
    print("THE SHAPE OF THE ORIGINAL GRAPH IS ", img.shape)
    img_valid = MyCorrelation(img, kernel, "valid")
    print("THE SHAPE OF THE MODIFIED GRAPH IS ", img_valid.shape)
    img_same = MyCorrelation(img, kernel, "same")
    print("THE SHAPE OF THE MODIFIED GRAPH IS ", img_same.shape)
    img_full = MyCorrelation(img, kernel, "full")
    print("THE SHAPE OF THE MODIFIED GRAPH IS ", img_full.shape)

    # (b) test reversing kernels
    in_order_kernel = np.array([[1/9, 2/9, 3/9], [2/9, 3/9, 4/9], [3/9, 4/9, 5/9]])
    reversed_kernel = reverse_convolution(in_order_kernel)
    print(reversed_kernel)

    # (c) portrait mode
    img = cv2.imread('mona_lisa.jpg')
    plt.imshow(img, cmap='gray')
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    kernel_3x3 = np.ones((3, 3), np.float32) / 9.0
    kernel_5x5 = np.ones((5, 5), np.float32) / 25.0
    img2[:, :, 0] = MyCorrelation(img2[:, :, 0], kernel_5x5, "same")
    img2[:, :, 1] = MyCorrelation(img2[:, :, 1], kernel_5x5, "same")
    img2[:, :, 2] = MyCorrelation(img2[:, :, 2], kernel_5x5, "same")
    img3 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # select 100:250 as the region of portrait
    img2[:, 100:250, 0] = img3[:, 100:250, 0]
    img2[:, 100:250, 1] = img3[:, 100:250, 1]
    img2[:, 100:250, 2] = img3[:, 100:250, 2]
    plt.imshow(img2)
    plt.show()
