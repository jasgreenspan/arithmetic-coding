from arithmetic_encoding import *
from golomb_encoding import *
import cv2


POSSIBLE_VALS = 2048


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
    int_lists = np.zeros((row_size, col_size, 64))
    for row in range(row_size):
        for col in range(col_size):
            block = blocked_img[row, col]
            coeff_mat = dct2(block)
            quantized_block = quantize_block(coeff_mat, b)

            int_list, _ = make_intlist_and_runlist(zigzag(quantized_block))

            int_lists[row, col, :len(int_list)] = int_list
            blocked_img[row, col] = quantized_block

    return blocked_img, int_lists


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


def coeff_vec_to_eq_vec(zigzags, k1, k2, k3, prev_first_coeff, with_arithmetic=False):
    """
    Encode the coefficient vector as follows:
        1. Number of runs to follow: Golomb-Rice of order k1
        2. Runs up to the last non-zero element: Golomb-Rice of order k2
        3. Integers: The first uncoded. the others with Exp-Golomb of order k3 + one sign bit
    :param prev_first_coeff: prev first coefficient for prediction
    :param k4: parameter for prediction encoding
    :param coeff_vec:
    :return:
    """
    int_list, run_list = make_intlist_and_runlist(zigzags)

    enc_num_of_runs = [encode_golomb_rice(len(run_list), k1)]
    enc_run_list = [encode_golomb_rice(x, k2) for x in run_list]

    first_coeff = int_list[0]
    # First uncoded coefficient: The number of bits used is always log2 of the number of possibilities, rounded up.
    sgn_bit = '0' if first_coeff < 0 else '1'
    num_of_bits = np.ceil(np.log2(POSSIBLE_VALS)).astype('int')
    encoded_first_coefficient = sgn_bit + format(abs(first_coeff), "0" + str(num_of_bits) + "b")

    enc_int_list_golomb = ''.join([encoded_first_coefficient] + [encode_exp_golomb(x, k3) for x in int_list[1:]])
    enc_int_list = enc_int_list_golomb
    if with_arithmetic:
        enc_int_list_arithmetic = compress_numbers_lst(int_list, state)
        if len(enc_int_list_arithmetic) < len(enc_int_list_golomb):
            print("Arithmetic code saved %d bits" % (len(enc_int_list_golomb) - len(enc_int_list_arithmetic)))
            enc_int_list = enc_int_list_arithmetic

    eq_vec = ''.join(enc_num_of_runs + enc_run_list) + enc_int_list
    return eq_vec, prev_first_coeff


def eq_vec_to_coeff_vec(eq_vec, start_idx, k1, k2, k3, prev_first_int, with_arithmetic=False):
    """
    Converts equivalent vector back to coefficient vector
    :return:
    """
    idx = start_idx

    # decodes the run_list
    run_len, idx = decode_golomb_rice(eq_vec, k1, idx)
    run_list = []
    for i in range(run_len):
        run,idx = decode_golomb_rice(eq_vec, k2, idx)
        run_list.append(run)

    # decodes the int_list
    temp_idx = idx
    sign = -1 if eq_vec[temp_idx] == '0' else 1
    temp_idx += 1
    num_of_bits = np.ceil(np.log2(POSSIBLE_VALS)).astype('int')
    first_int, temp_idx = decode_binary_string(eq_vec, temp_idx, num_of_bits) * sign

    int_list = [first_int]

    for i in range(run_len):
        num, temp_idx = decode_exp_golomb(eq_vec, k3, temp_idx)
        int_list.append(num)

    if with_arithmetic:
        int_list_length = run_len + 1
        idx_before_lst = idx
        temp_int_list, idx = decompress_numbers_lst(eq_vec, state, int_list_length, idx)
        if idx_before_lst - idx < temp_idx - idx_before_lst: # Arithmetic coding was used
            int_list = temp_int_list
        else:
            idx = temp_idx

    coeff_vec = recreate_coeff_vec(run_list, int_list, BLOCK_SIZE)
    return coeff_vec, idx, prev_first_int


def recreate_coeff_vec(run_list, int_list, n):
    """
    Puts the ints and zeroes in proper order in coefficient vector to recreate it
    :param orig_m:
    :param orig_n:
    :param run_list:
    :param int_list:
    :return:
    """
    coeff_vec = []
    for idx, num_zeroes in enumerate(run_list):
        coeff_vec.append(int_list[idx])
        coeff_vec += [0] * num_zeroes
    coeff_vec.append(int_list[-1])

    last_run = (n**2) - len(coeff_vec)
    coeff_vec += [0] * last_run
    return coeff_vec


def make_intlist_and_runlist(zigzags):
    int_list, run_list, run_counter = [], [], 0

    first, zigzags = zigzags[0], zigzags[1:]
    int_list.append(first)

    for num in zigzags:
        if num == 0:
            run_counter += 1
        else:
            int_list.append(num)
            run_list.append(run_counter)
            run_counter = 0

    return int_list, run_list


def code_to_image(encoding, k1, k2, k3, b, with_arithmetic=False):
    """
    Convert coefficient vector to img
    :param encoding:
    :return: img
    """
    prev_first_int = 0
    new_m = orig_m // BLOCK_SIZE
    new_n = orig_n // BLOCK_SIZE
    binned_matrix = np.empty((new_m, new_n, BLOCK_SIZE, BLOCK_SIZE))

    # First decode state
    idx = 0
    state_encoding_len, idx = decode_binary_string(encoding, idx, FRACTION_ENC_LENGTH)
    encoded_state = encoding[idx: idx + state_encoding_len]
    idx += state_encoding_len
    state = StateMachine(encoded_state)

    # Create blocks
    for row in range(new_m):
        for col in range(new_n):
            # 1. Decode slice
            decoded_slice, idx, prev_first_int = eq_vec_to_coeff_vec(encoding, idx, k1, k2, k3, prev_first_int,
                                                                     with_arithmetic)
            # 2. Inverse zig zag
            mat = inv_zigzag(np.array(decoded_slice))
            # 3. De-quantize
            dequantize_mat = mat * b
            # 4. IDCT
            inv_mat = idct2(dequantize_mat)
            binned_matrix[row][col] = inv_mat

    # 4. Deblock
    a = deblock_img(binned_matrix)
    # 5. Unscale image
    return unscale(a)


def quantized_image_to_vec(blocked_img, k1, k2, k3, with_arithmetic=False):
    """
    Convert img to coefficient vector
    :param img:
    :return: coefficient vector
    """
    zigzags = []
    prev_first_coeff = 0

    row_size, col_size, dim, dim = blocked_img.shape
    for row in range(row_size):
        for col in range(col_size):
            print("Encoding block %d %d" % (row, col))
            quantized_block = blocked_img[row, col]
            coeff_vec = zigzag(quantized_block)
            eq_vec, prev_first_coeff = coeff_vec_to_eq_vec(coeff_vec, k1, k2, k3, prev_first_coeff, with_arithmetic)
            zigzags += [eq_vec]

    return ''.join(zigzags)

if __name__ == '__main__':
    quantization_param = 50
    order1, order2, order3 = 3, 2, 2

    # Read image, and preprocess by DCT and quantization
    a = cv2.imread('Mona-Lisa.bmp', cv2.IMREAD_GRAYSCALE)
    orig_m, orig_n = a.shape
    quantized_a, coefficients = quantize_img(a, quantization_param)

    # Encode image using only Exp-Golomb Encoding
    golomb_code = quantized_image_to_vec(quantized_a, order1, order2, order3, with_arithmetic=False)
    total_golomb_encoding_len = len(golomb_code)

    # Encode image with Arithmetic Coding
    state = StateMachine(coefficients)
    encoded_state = state.encode_state()
    arithmetic_code = quantized_image_to_vec(quantized_a, order1, order2, order3, with_arithmetic=True)
    total_arithmetic_coding = format(len(encoded_state), FRACTION_ENC) + encoded_state + arithmetic_code
    total_arithmetic_coding_len = len(total_arithmetic_coding)

    print("Encoded image using Order %d Exp-Golomb using %d bits" % (order3, total_golomb_encoding_len))
    print("Encoded image using Arithmetic Coding using %d bits" % total_arithmetic_coding_len)

    # Decode image with Arithmetic Coding and compare with original
    decoded = code_to_image(arithmetic_code, order1, order2, order3, quantization_param)

    plt.imshow(a, cmap='gray')
    plt.title("Original Image")
    plt.show()
    plt.imshow(decoded, cmap='gray')
    plt.title("Decompressed Image")
    plt.show()

    print((a == decoded).all())

