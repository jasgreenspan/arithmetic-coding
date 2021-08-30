from utils import *
from numpy.lib.stride_tricks import as_strided as ast
import cv2


if __name__ == '__main__':
    # TODO: make comparison scheme
    b = 50
    a = cv2.imread('Mona-Lisa.bmp', cv2.IMREAD_GRAYSCALE)
    quantize_a = quantize_img(a)
    get_distribution(quantize_a)
