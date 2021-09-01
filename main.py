from arithmetic_encoding import *
from golomb_encoding import *
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


def quantize_block(block, b):
    """
    Perform uniform quantization on a block of DCT values
    :param block: the DCT values
    :param b: the quantization parameter
    :return:
    """
    # Quantization of each coefficient separately, using a uniform quantizer of cell length b
    min_val = np.min(block)
    max_val = np.max(block)
    num_bins, a_min, a_max = find_num_bins(max_val, min_val, b)  # a_min is the 'a' value of the lowest bin,
                                                                 # a_max is the 'a' value of the highest bin
    half_b = b / 2
    lower_bin_limit = a_min * half_b
    upper_bin_limit = a_max * half_b
    bins = np.linspace(lower_bin_limit, upper_bin_limit, int(num_bins))

    # Sort the coefficients into the bins and recenter them around 0
    return (np.digitize(block, bins) + ((a_min + 1) / 2) - 1).astype(int)


def quantize_img(img, b):
    """
    Perform DCT on the image and quantize the values
    :param img: the image to be processed
    :param b: the quantization parameter
    :return: the quantized image
    """
    scaled_img = scaling(img) # Scaling to reals centered at 0
    blocked_img = block_img(scaled_img)

    row_size, col_size, dim, dim = blocked_img.shape
    for row in range(row_size):
        for col in range(col_size):
            block = blocked_img[row, col]
            coeff_mat = dct2(block)
            quantized_block = quantize_block(coeff_mat, b)
            blocked_img[row, col] = quantized_block

    quantized_img = deblock_img(blocked_img)
    return quantized_img


def dequantize_img(quantized_img, b):
    """
    Reverse the process of quantization and DCT on the image
    :param quantized_img: the image to be processed
    :param b: the quantization parameter
    :return:
    """
    blocked_img = block_img(quantized_img)

    row_size, col_size, dim, dim = blocked_img.shape
    for row in range(row_size):
        for col in range(col_size):
            block = blocked_img[row, col]
            coeff_mat = block * b
            orig_block = idct2(coeff_mat)
            blocked_img[row, col] = orig_block

    deblocked_img = deblock_img(blocked_img)
    dequantized_image = unscale(deblocked_img)
    return dequantized_image


if __name__ == '__main__':
    quantization_param = 50
    a = cv2.imread('Mona-Lisa.bmp', cv2.IMREAD_GRAYSCALE)[:128, :128]
    a = quantize_img(a, quantization_param)

    # Calculate length of encoding image with Exp-Golomb Encoding
    golomb_encoding_len_by_pixel = np.vectorize(exp_golomb_length)(a, GOLOMB_ENC_ORDER)
    total_golomb_encoding_len = np.sum(golomb_encoding_len_by_pixel)

    # Encode image with Arithmetic Coding
    state = StateMachine(a)
    code = compress_image(a, state)
    total_arithmetic_coding_len = len(code)

    print("Encoded image using Order %d Exp-Golomb using %d bits" % (GOLOMB_ENC_ORDER, total_golomb_encoding_len))
    print("Encoded image using Arithmetic Coding using %d bits" % total_arithmetic_coding_len)

