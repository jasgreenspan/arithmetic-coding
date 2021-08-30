from utils import *
from fractions import Fraction
from ctypes import c_uint32 as unsigned_byte
import cv2
import matplotlib.pyplot as plt
import math

BIT_LENGTH = 32
FRACTION_ENC_LENGTH = 16
FRACTION_ENC = "0%db" % FRACTION_ENC_LENGTH

class StateMachine:
    def __init__(self, img):
        self.scaled_total = None
        self.intervals = self._get_intervals(img)
        self.shape = img.shape

    def _get_rational_distribution(self, img):
        """
        Get the distribution of values in the image
        :param img: the image
        :return: dictionary of values and probabilities
        """
        values, counts = np.unique(img, return_counts=True)
        prob = np.array([values, counts], dtype=np.float).T
        prob[:, 1] /= img.size
        distribution = dict(zip(prob[:, 0], prob[:, 1]))
        return distribution

    def _get_integer_distribution(self, img):
        """
        Using Michael Dipperstein's algorithm.
        Accessed August 2021: http://michael.dipperstein.com/arithmetic/index.html
        :param img:
        :return:
        """
        values, counts = np.unique(img, return_counts=True)
        prob = np.array([values, counts]).T

        # Scale the probability range
        # 1. Divide the total symbol count by 2^(N - 2) and ceil
        scaled_total = math.ceil(img.size / (2 ** (BIT_LENGTH - 2)))
        # 2. Using integer division, divide each individual symbol count by the above.
        prob[:, 1] //= scaled_total
        scaled_zero = (prob[:, 1] == 0)
        # If a non-zero count became zero, make it one.
        prob[scaled_zero] = 1
        self.scaled_total = np.sum(prob[:, 1])
        distribution = dict(zip(prob[:, 0], prob[:, 1]))
        return distribution

    # Find value interval
    def _get_intervals(self, img):
        """
        Calculate initial intervals for arithmetic encoding
        :param img:
        :return: a dictionary where key: image value --> (probability, start of range, end of range)
        """
        d = self._get_rational_distribution(img)
        distribution = sorted(d.items(), key=lambda x: x[1], reverse=True)
        intervals = {}
        lower = 0

        for val, prob in distribution:
            upper = lower + prob
            intervals[val] = (Fraction(str(lower)), Fraction(str(upper)))
            lower += prob

        return intervals


def compress_image(img, state):
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
            upper_bound, lower_bound = encode_symbol(val, state, lower_bound, upper_bound)

        # Convert sub-interval to binary encoding
        midway_point = (lower_bound + (upper_bound - lower_bound) / 2)
        encoded_numerator = bin(midway_point.numerator)[2:]
        encoded_denominator = bin(midway_point.denominator)[2:]
        encoded_block = format(len(encoded_numerator), FRACTION_ENC) + format(len(encoded_denominator), FRACTION_ENC) \
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

    return encoded_img


def output_bit_plus_pending(bit, pending_bits):
    output_seq = str(bit)
    while pending_bits > 0:
        output_seq += "1" if bit == 0 else "0"
        pending_bits -= 1

    print(output_seq, end=" ")
    return output_seq


def encode_symbol(symbol, state, lower_bound, upper_bound):
    """
    Based on "Arithmetic Coding for Data Compression", Witten, Neal, and Cleary (1987)
    Accessed August 2021 from:
    https://www.researchgate.net/publication/200065260_Arithmetic_Coding_for_Data_Compression
    :param symbol:
    :return:
    """
    orig_low, orig_high = state.intervals[symbol.item()]
    range = upper_bound - lower_bound
    high = lower_bound + range * orig_high
    low = lower_bound + range * orig_low

    return high, low


def decode_symbol(orig_low, orig_high, lower_bound, upper_bound):
    """
    :return:
    """
    range = upper_bound - lower_bound
    high = lower_bound + range * orig_high
    low = lower_bound + range * orig_low

    return high, low


def decode_image(encoding, state):
    # TODO: encode state
    m, n = state.shape
    new_m, new_n = m // BLOCK_SIZE, n // BLOCK_SIZE
    num_of_blocks = new_m * new_n
    decoded_img = np.empty((num_of_blocks, BLOCK_SIZE, BLOCK_SIZE))
    idx = 0

    # Decode each block separately using arithmetic coding
    for i in range(num_of_blocks):
        print("Decoding %d 'th block" % i)
        # Decode the binary string representing the block
        num_len = int(encoding[idx:idx + FRACTION_ENC_LENGTH], 2)
        idx += FRACTION_ENC_LENGTH
        den_len = int(encoding[idx:idx + FRACTION_ENC_LENGTH], 2)
        idx += FRACTION_ENC_LENGTH
        decoded_numerator = int(encoding[idx:idx + num_len], 2)
        idx += num_len
        decoded_denominator = int(encoding[idx:idx + den_len], 2)
        idx += den_len

        # Reverse the arithmetic coding by going over all the ranges
        block_result = []
        midway_point = Fraction(decoded_numerator, decoded_denominator)
        lower_bound = 0
        upper_bound = 1

        while len(block_result) < BLOCK_SIZE ** 2:
            subinterval = upper_bound - lower_bound
            for orig_val in state.intervals.keys():
                # The next encoded symbol has the point in its range
                orig_low, orig_high = state.intervals[orig_val]
                # Check if symbol is encoded in inverse of orig. calc. low = lower_bound + range * orig_low
                if orig_low <= (midway_point - lower_bound) / subinterval < orig_high:
                    block_result += [orig_val]
                    upper_bound, lower_bound = decode_symbol(orig_low, orig_high, lower_bound, upper_bound)

        decoded_img[i] = np.asarray(block_result).reshape(BLOCK_SIZE, BLOCK_SIZE)

    decoded_img = deblock_img(decoded_img.reshape((new_m, new_n, BLOCK_SIZE, BLOCK_SIZE)))
    return decoded_img


if __name__ == '__main__':
    # a = np.arange(8)
    # b = np.ones((8, 8), dtype=np.int32)
    # a = a * b
    a = cv2.imread('Mona-Lisa.bmp', cv2.IMREAD_GRAYSCALE)[:128, :128]
    a = cv2.imread('Mona-Lisa.bmp', cv2.IMREAD_GRAYSCALE)
    state = StateMachine(a)
    code = compress_image(a, state)
    decoded = decode_image(code, state)

    plt.imshow(a, cmap='gray')
    plt.show()
    plt.imshow(decoded, cmap='gray')
    plt.show()

    print((a == decoded).all())
