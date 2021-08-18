from utils import *
from numpy.lib.stride_tricks import as_strided as ast
import cv2

def find_num_bins(max_val, min_val, b):
    """
    Finds the number of bins
    :param max_val:
    :param min_val:
    :param b:
    :return:
    """
    min_bin_idx, a_min = find_bin_idx(min_val, b)
    max_bin_idx, a_max = find_bin_idx(max_val, b)
    num_bins = abs(max_bin_idx - min_bin_idx) + 1

    return num_bins, a_min, a_max


def find_bin_idx(num, b):
    """
    Each number in existence is between a(b/2) and (a+2)(b/2) for some 'a' and for a given b
    :param num:
    :param b:
    :return:
    """
    half_b = b / 2
    bounded_by_a = np.floor(num / half_b)
    if bounded_by_a % 2 == 0:
        bounded_by_a -= 1
    a = bounded_by_a
    bin_index = (half_b * (a + 1)) / b
    return bin_index, a


def quantize_img(img):
    coeff_mat = dct2(img)
    # 4. Quantization of each coefficient separately, using a uniform quantizer of cell length b
    min_val = np.min(coeff_mat)
    max_val = np.max(coeff_mat)
    num_bins, a_min, a_max = find_num_bins(max_val, min_val, b) # a_min is the 'a' value of the lowest bin,
    # a_max is the 'a' value of the highest bin
    half_b = b / 2
    lower_bin_limit = a_min * half_b
    upper_bin_limit = a_max * half_b
    bins = np.linspace(lower_bin_limit, upper_bin_limit, int(num_bins))

    # Sort the coefficients into the bins and recenter them around 0
    quantize_idx = (np.digitize(coeff_mat, bins)+((a_min+1)/2)-1).astype(int)
    return quantize_idx


if __name__ == '__main__':
    b = 50
    a = cv2.imread('Mona-Lisa.bmp', cv2.IMREAD_GRAYSCALE)
    quantize_a = quantize_img(a)
    get_distribution(quantize_a)
