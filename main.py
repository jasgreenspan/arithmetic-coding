from arithmetic_encoding import *
from golomb_encoding import *
import cv2


POSSIBLE_VALS = 2048
GOLOMB_USED = "0"
ARITHMETIC_USED = "1"
FULL_EXP_GOLOMB_MODE = 0
FULL_ARITHMETIC_MODE = 1
HYBRID_GOLOMB_ARITHMETIC_MODE = 2


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


def coeff_vec_to_eq_vec(zigzags, k1, k2, k3, prev_first_coeff, mode):
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
    if mode == FULL_ARITHMETIC_MODE:
        enc_int_list = compress_numbers_lst(int_list, state)
    elif mode == HYBRID_GOLOMB_ARITHMETIC_MODE:
        enc_int_list = GOLOMB_USED + enc_int_list # Encode the choice of encoding
        enc_int_list_arithmetic = compress_numbers_lst(int_list, state)
        if len(enc_int_list_arithmetic) < len(enc_int_list_golomb):
            enc_int_list = ARITHMETIC_USED + enc_int_list_arithmetic

    eq_vec = ''.join(enc_num_of_runs + enc_run_list) + enc_int_list
    return eq_vec, prev_first_coeff


def decode_golomb_int_list(eq_vec, run_len, k3, idx):
    sign = -1 if eq_vec[idx] == '0' else 1
    idx += 1
    num_of_bits = np.ceil(np.log2(POSSIBLE_VALS)).astype('int')
    first_int, idx = decode_binary_string(eq_vec, idx, num_of_bits)
    first_int *= sign

    int_list = [first_int]

    for i in range(run_len):
        num, idx = decode_exp_golomb(eq_vec, k3, idx)
        int_list.append(num)

    return int_list, idx


def eq_vec_to_coeff_vec(eq_vec, start_idx, k1, k2, k3, prev_first_int, decoded_state, mode):
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
    if mode == FULL_ARITHMETIC_MODE:
        int_list_length = run_len + 1
        int_list, idx = decompress_numbers_lst(eq_vec, decoded_state, int_list_length, idx)
    elif mode == HYBRID_GOLOMB_ARITHMETIC_MODE:
        indicator_bit = eq_vec[idx]
        idx += 1

        if indicator_bit == ARITHMETIC_USED:
            int_list_length = run_len + 1
            int_list, idx = decompress_numbers_lst(eq_vec, decoded_state, int_list_length, idx)
        else:
            int_list, idx = decode_golomb_int_list(eq_vec, run_len, k3, idx)
    else:
        int_list, idx = decode_golomb_int_list(eq_vec, run_len, k3, idx)

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


def code_to_image(encoding, k1, k2, k3, b, mode=FULL_EXP_GOLOMB_MODE):
    """
    Convert coefficient vector to img
    :param encoding:
    :return: img
    """
    prev_first_int = 0
    new_m = orig_m // BLOCK_SIZE
    new_n = orig_n // BLOCK_SIZE
    binned_matrix = np.empty((new_m, new_n, BLOCK_SIZE, BLOCK_SIZE))

    idx = 0
    decoded_state = None
    if mode == FULL_ARITHMETIC_MODE or mode == HYBRID_GOLOMB_ARITHMETIC_MODE:
        # First decode state
        state_encoding_len, idx = decode_binary_string(encoding, idx, FRACTION_ENC_LENGTH)
        encoded_state = encoding[idx: idx + state_encoding_len]
        idx += state_encoding_len
        decoded_state = StateMachine(encoded_state)

    # Create blocks
    for row in range(new_m):
        for col in range(new_n):
            # 1. Decode slice
            decoded_slice, idx, prev_first_int = eq_vec_to_coeff_vec(encoding, idx, k1, k2, k3, prev_first_int,
                                                                     decoded_state, mode)
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


def quantized_image_to_vec(blocked_img, k1, k2, k3, mode=FULL_EXP_GOLOMB_MODE):
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
            quantized_block = blocked_img[row, col]
            coeff_vec = zigzag(quantized_block)
            eq_vec, prev_first_coeff = coeff_vec_to_eq_vec(coeff_vec, k1, k2, k3, prev_first_coeff, mode)
            zigzags += [eq_vec]

    return ''.join(zigzags)

if __name__ == '__main__':
    quantization_param = 50
    order1, order2, order3 = 3, 2, 2
    test_images = ['Mona-Lisa.bmp', 'lena_512.tif', 'lena_256.tif', 'woman_blonde.tif', 'woman_darkhair.tif']
    algorithms = ["Full Exp-Golomb", "Full Arithmetic Coding", "Hybrid Golomb Arithmetic"]
    bpps = [[], [], []]

    for test_image in test_images:
        # Read image, and preprocess by DCT and quantization
        a = cv2.imread(test_image, cv2.IMREAD_GRAYSCALE)
        orig_m, orig_n = a.shape
        quantized_a, coefficients = quantize_img(a, quantization_param)

        # Encode image using only Exp-Golomb Encoding
        golomb_code = quantized_image_to_vec(quantized_a, order1, order2, order3)
        total_golomb_encoding_len = len(golomb_code)

        # Encode image with Arithmetic Coding
        state = StateMachine(coefficients)
        encoded_state = state.encode_state()
        full_arithmetic_code = quantized_image_to_vec(quantized_a, order1, order2, order3, mode=FULL_ARITHMETIC_MODE)
        hybrid_arithmetic_code = quantized_image_to_vec(quantized_a, order1, order2, order3, mode=HYBRID_GOLOMB_ARITHMETIC_MODE)
        total_full_arithmetic_coding = format(len(encoded_state), FRACTION_ENC) + encoded_state + full_arithmetic_code
        total_hybrid_arithmetic_coding = format(len(encoded_state), FRACTION_ENC) + encoded_state + hybrid_arithmetic_code

        print("Encoded state for %s using %d bits" % (test_image, len(encoded_state)))
        print("Encoded %s using Order %d Exp-Golomb using %d bits" % (test_image, order3, total_golomb_encoding_len))
        bpps[0] += [total_golomb_encoding_len / (orig_m * orig_n)]
        print("Encoded %s using Full Arithmetic Coding using %d bits" % (test_image, len(total_full_arithmetic_coding)))
        bpps[1] += [len(total_full_arithmetic_coding) / (orig_m * orig_n)]
        print("Encoded %s using Hybrid Arithmetic Coding using %d bits" % (test_image, len(total_hybrid_arithmetic_coding)))
        bpps[2] += [len(total_hybrid_arithmetic_coding) / (orig_m * orig_n)]

        # Decode image with Arithmetic Coding and compare with original
        decoded_full = code_to_image(total_full_arithmetic_coding, order1, order2, order3, quantization_param, mode=FULL_ARITHMETIC_MODE)
        decoded_hybrid = code_to_image(total_hybrid_arithmetic_coding, order1, order2, order3, quantization_param, mode=HYBRID_GOLOMB_ARITHMETIC_MODE)

        plt.imshow(a, cmap='gray')
        plt.title("Original Image")
        plt.show()
        plt.imshow(decoded_full, cmap='gray')
        plt.title("Decompressed Image")
        plt.show()

    make_bar_graph(algorithms, test_images, bpps)
