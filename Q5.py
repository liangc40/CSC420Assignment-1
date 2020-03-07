import numpy as np


def isSeparable(kernel):
    """
    Check whether a kernel is separable or not.
    :param kernel: kernel represented in matrix form
    :type kernel: [[int]]
    :return: a boolean indicating whether the input kernel is separable or not,
             and two [int] representing the 1-D kernels whose product equals the input kernel
    :rtype: boolean, [int], [int]
    """
    sep = False
    U, S, V = np.linalg.svd(kernel)
    kernel_V = U[:, 0] * np.sqrt(S[0])
    kernel_h = V[0] * np.sqrt(S[0])
    if np.linalg.matrix_rank(kernel) == 1:
        sep = True
    return sep, kernel_V, kernel_h


if __name__ == '__main__':
    # (b) test cases to show whether the input kernel is separable or not.
    # if it could be broken into two 1-D kernels, print these two kernels.
    non_sep_example_one = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    non_sep_example_two = np.array([[-1, 2, 0, 4], [0, 7, 1, -12], [0, 0, 0, 0], [0, 0, 0, 0]])
    sep_example_one = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    sep_example_two = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    example_list = [non_sep_example_one, non_sep_example_two, sep_example_one, sep_example_two]

    for i in range(len(example_list)):
        separable, h1, h2 = isSeparable(example_list[i])
        if separable is False:
            print("FOR CASE ", i, " THE MATRIX IS NOT SEPARABLE")
        else:
            print("FOR CASE ", i, " THE MATRIX IS SEPARABLE, AND COULD BE SEPARATED INTO", h1, " AND", h2)
