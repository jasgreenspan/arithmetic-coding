from utils import *
from golomb_encoding import encode_exp_golomb, decode_exp_golomb
from fractions import Fraction
import cv2
import matplotlib.pyplot as plt

BIT_LENGTH = 8
GOLOMB_ENC_ORDER = 8
PROB_GOLOMB_ORDER = 7
BOUNDS_LEN_GOLOMB_ORDER = 9
SHAPE_ENC_LENGTH = 16
SHAPE_ENC = "0%db" % SHAPE_ENC_LENGTH
VALS_ENC_LENGTH = 8
FRACTION_ENC_LENGTH = 16
FRACTION_ENC = "0%db" % FRACTION_ENC_LENGTH

class StateMachine:
    def __init__(self, arg):
        # Initialize StateMachine from encoded string
        if isinstance(arg, str):
            idx = 0
            self.shape, idx = self._decode_shape(arg, idx)
            self.distribution, idx = self._decode_distribution(arg, idx)
        else: # Initialize StateMachine from image
            self.shape = arg.shape
            self.distribution = self._get_distribution_by_binary(arg)

        self.intervals = self._get_intervals(self.distribution)

    def _get_distribution_by_value(self, img):
        """
        Get the distribution of values (either pixel or DCT) in the image
        :param img: the image
        :return: dictionary of values and probabilities
        """
        # Calculate the probabilities by counting how many times each value appears and dividing by total
        values, counts = np.unique(img, return_counts=True)
        prob = np.array([values, counts], dtype=np.float).T
        prob[:, 1] /= img.size

        # Return a dictionary of {value in image : probability of value appearing}
        distribution = dict(zip(prob[:, 0], prob[:, 1]))
        return distribution

    def _get_distribution_by_binary(self, img):
        """
        Get the distribution of 0's and 1's in the binary representing of the image
        :param img: the image
        :return: dictionary of values and probabilities
        """
        # Convert each element of the image to its binary representation
        img_in_binary = np.vectorize(np.binary_repr)(img, width=BIT_LENGTH).flatten()
        img_as_binary_str = ''.join(img_in_binary.tolist())

        # Calculate the probabilities by counting how many times each value appears and dividing by total
        distribution = {}
        distribution["0"] = img_as_binary_str.count("0") / len(img_as_binary_str)
        distribution["1"] = img_as_binary_str.count("1") / len(img_as_binary_str)

        return distribution

    def _get_intervals(self, distribution):
        """
        Calculate initial intervals for arithmetic encoding
        :param distribution: the distribution of values in the image
        :return: a dictionary where key: image value --> (start of range, end of range)
        """
        d = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        intervals = {}
        lower = 0

        # Divide up the interval [0,1) to sub-intervals representing each value, from largest to smallest
        for val, prob in d:
            upper = lower + prob
            intervals[val] = (Fraction(str(lower)), Fraction(str(upper)))
            lower += prob

        return intervals

    def encode_state(self):
        """
        Encode the field values of the state in a binary string that can be used in the constructor
        :return: the encoded string
        """
        encoded_shape = self._encode_shape()
        encoded_distribution = self._encode_distribution()

        return encoded_shape + encoded_distribution

    def _encode_shape(self):
        """
        Encode the shape
        """
        m, n = self.shape
        enc_m = format(m, SHAPE_ENC)
        enc_n = format(n, SHAPE_ENC)

        return enc_m + enc_n

    def _encode_distribution(self):
        """
        Encode the distribution
        """
        encoding = ""

        for val, prob in self.distribution.items():
            prob = Fraction(prob)
            enc_val = encode_exp_golomb(int(val), GOLOMB_ENC_ORDER)
            enc_prob_numerator = encode_exp_golomb(prob.numerator, PROB_GOLOMB_ORDER)
            enc_prob_denominator = encode_exp_golomb(prob.denominator, PROB_GOLOMB_ORDER)

            encoding += enc_val + enc_prob_numerator + enc_prob_denominator

        return encoding

    def _decode_shape(self, encoding, idx):
        """
        Recreate the image shape from an encoding
        """
        m, idx = decode_binary_string(encoding, idx, SHAPE_ENC_LENGTH)
        n, idx = decode_binary_string(encoding, idx, SHAPE_ENC_LENGTH)

        shape = (m, n)
        return shape, idx

    def _decode_distribution(self, encoding, idx):
        """
        Recreate the distribution of values from an encoded string
        :param encoding: the encoded string
        :param idx: current index in the string
        :return: dictionary of values and probabilities
        """
        distribution = {}
        while idx < len(encoding):
            val, idx = decode_exp_golomb(encoding, GOLOMB_ENC_ORDER, idx)
            prob_numerator, idx = decode_exp_golomb(encoding, PROB_GOLOMB_ORDER, idx)
            prob_denominator, idx = decode_exp_golomb(encoding, PROB_GOLOMB_ORDER, idx)

            distribution[val] = Fraction(prob_numerator, prob_denominator)

        return distribution, idx


def decode_binary_string(binary_str, current_idx, encoding_length):
    """
    Helper function to decode an integer encoded in binary with different lengths
    :param binary_str: the full string
    :param current_idx: the current index in the full string
    :param encoding_length: the length of the encoding
    :return: the integer and the new index
    """
    val = int(binary_str[current_idx:current_idx + encoding_length], 2)
    new_idx = current_idx + encoding_length

    return val, new_idx


def compress_image(img, state):
    """
    Given an image and a frequency table, encode the image using arithmetic coding
    Based on "Arithmetic Coding for Data Compression", Witten, Neal, and Cleary (1987)
    Accessed August 2021 from:
    https://www.researchgate.net/publication/200065260_Arithmetic_Coding_for_Data_Compression
    :param img: the image to be encoded (either in pixel or DCT values)
    :param state: the state representing the frequency of the values to be encoded
    :return: a binary string representing the encoded image
    """
    blocks = block_img(img).reshape(-1, BLOCK_SIZE, BLOCK_SIZE)
    encoded_img = ''
    counter = 0

    # Encode each block separately using arithmetic coding
    for block in blocks:
        print("Encoding %d 'th block" % counter)
        counter += 1
        lower_bound = 0
        upper_bound = 1

        # Find sub-interval to encode block
        for val in np.nditer(block):
            val_in_binary = np.binary_repr(val, width=BIT_LENGTH)
            for bit in val_in_binary:
                upper_bound, lower_bound = encode_symbol(bit, state, lower_bound, upper_bound)
            # upper_bound, lower_bound = encode_symbol(val, state, lower_bound, upper_bound)

    # Convert sub-interval to binary encoding
    midway_point = (lower_bound + (upper_bound - lower_bound) / 2)
    encoded_numerator = bin(midway_point.numerator)[2:]
    encoded_denominator = bin(midway_point.denominator)[2:]
    encoded_block = encode_exp_golomb(len(encoded_numerator), BOUNDS_LEN_GOLOMB_ORDER) \
                    + encode_exp_golomb(len(encoded_denominator), BOUNDS_LEN_GOLOMB_ORDER) \
                    + encoded_numerator + encoded_denominator

    # encoded_block = "0.1"
    # while True:
    #     code_as_fraction = Fraction(str(convert_to_decimal_fraction(encoded_block)))
    #     # When to left of lower bound, make LSB '1'
    #     if code_as_fraction < lower_bound:
    #         encoded_block += "1"
    #     # Check if binary fraction is in range, if so break
    #     if upper_bound > code_as_fraction >= lower_bound:
    #         break
    #     # When to right of upper bound, make LSB '0'
    #     if code_as_fraction > upper_bound:
    #         encoded_block = encoded_block[:-1] + "0"

    encoded_img += encoded_block

    # Encode the state
    encoded_state = state.encode_state()
    full_encoding = format(len(encoded_state), FRACTION_ENC) + encoded_state + encoded_img

    return full_encoding


def encode_symbol(symbol, state, lower_bound, upper_bound):
    """
    Finds the new range after encoding the given symbol
    :param symbol: the symbol/value current being encoded
    :param state: the state representing the distribution of values in the image
    :param lower_bound: the current lower bound of the range
    :param upper_bound: the current upper bound of the range
    :return:
    """
    orig_low, orig_high = state.intervals[symbol]
    range = upper_bound - lower_bound
    high = lower_bound + range * orig_high
    low = lower_bound + range * orig_low

    return high, low


def decode_symbol(orig_low, orig_high, lower_bound, upper_bound):
    """
    Recreates the range created by encoding a given symbol
    :param orig_low: the symbol's lower bound in the original distribution
    :param orig_high: the symbol's upper bound in the original distribution
    :param lower_bound: the current lower bound of the range
    :param upper_bound: the current upper bound of the range
    :return:
    """
    range = upper_bound - lower_bound
    high = lower_bound + range * orig_high
    low = lower_bound + range * orig_low

    return high, low


def decompress_image(encoding):
    """
    Given an image encoded using arithmetic coding and encoded frequency table, recreate the image
    Based on "Arithmetic Coding for Data Compression", Witten, Neal, and Cleary (1987)
    Accessed August 2021 from:
    https://www.researchgate.net/publication/200065260_Arithmetic_Coding_for_Data_Compression
    :param encoding: a binary string representing the encoded image
    :return: the decompressed image
    """
    # TODO account for binary probability scheme
    # TODO add binary probablity as parameter
    # First decode state
    idx = 0
    state_encoding_len, idx = decode_binary_string(encoding, idx, FRACTION_ENC_LENGTH)
    encoded_state = encoding[idx: idx + state_encoding_len]
    idx += state_encoding_len
    state = StateMachine(encoded_state)

    # Then decode image using the state
    m, n = state.shape
    new_m, new_n = m // BLOCK_SIZE, n // BLOCK_SIZE
    num_of_blocks = new_m * new_n
    decoded_img = np.empty((num_of_blocks, BLOCK_SIZE, BLOCK_SIZE))

    # Decode each block separately using arithmetic coding
    for i in range(num_of_blocks):
        print("Decoding %d 'th block" % i)
        # Decode the binary string representing the block
        num_len, idx = decode_exp_golomb(encoding, BOUNDS_LEN_GOLOMB_ORDER, idx)
        den_len, idx = decode_exp_golomb(encoding, BOUNDS_LEN_GOLOMB_ORDER, idx)
        decoded_numerator, idx = decode_binary_string(encoding, idx, num_len)
        decoded_denominator, idx = decode_binary_string(encoding, idx, den_len)

        # Reverse the arithmetic coding by going over all the ranges
        block_result = []
        midway_point = Fraction(decoded_numerator, decoded_denominator)
        lower_bound = 0
        upper_bound = 1

        while len(block_result) < BLOCK_SIZE ** 2:
            interval = upper_bound - lower_bound
            for orig_val, bounds in state.intervals.items():
                orig_low, orig_high = bounds

                # Find the symbol that would have been encoded in the current range
                # Using the inverse of original calculation: low = lower_bound + range * orig_low
                if orig_low <= (midway_point - lower_bound) / interval < orig_high:
                    block_result += [orig_val]
                    upper_bound, lower_bound = decode_symbol(orig_low, orig_high, lower_bound, upper_bound)
                    break

        decoded_img[i] = np.asarray(block_result).reshape(BLOCK_SIZE, BLOCK_SIZE)

    decoded_img = deblock_img(decoded_img.reshape((new_m, new_n, BLOCK_SIZE, BLOCK_SIZE)))
    return decoded_img


if __name__ == '__main__':
    a = cv2.imread('Mona-Lisa.bmp', cv2.IMREAD_GRAYSCALE)[:32, :32]
    # a = cv2.imread('Mona-Lisa.bmp', cv2.IMREAD_GRAYSCALE)
    state = StateMachine(a)
    code = compress_image(a, state)
    decoded = decompress_image(code)

    plt.imshow(a, cmap='gray')
    plt.show()
    plt.imshow(decoded, cmap='gray')
    plt.show()

    print((a == decoded).all())
