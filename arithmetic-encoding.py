from utils import *
import cv2
import matplotlib.pyplot as plt

TERMINTAOR = "{0:b}".format(65535)

class StateMachine:
    def __init__(self, img):
        self.intervals = self._get_intervals(img)
        self.shape = img.shape

    def _get_distribution(self, img):
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

    # Find value interval
    def _get_intervals(self, img):
        d = self._get_distribution(img)
        distribution = sorted(d.items(), key=lambda x: x[1], reverse=True)
        intervals = {}
        start = 0.0
        for val, prob in distribution:
            intervals[val] = (prob, start, start + prob)
            start += prob

        return intervals


def compress_image(img, state):
    blocks = block_img(img)
    encoded_img = ''

    # Encode each block separately using arithmetic coding
    for block in blocks:
        lower_bound = 0.0
        upper_bound = 1.0

        # Find sub-interval to encode block
        for val in np.nditer(block):
            upper_bound, lower_bound = encode_symbol(val, state, lower_bound, upper_bound)

        # Convert sub-interval to binary encoding
        bin_fraction = "0."
        while True:
            # When to left of lower bound, make LSB '1'
            bin_fraction += "1"
            if convert_to_decimal_fraction(bin_fraction) > lower_bound:
                break
            # When to right of upper bound, make LSB '0'
            if convert_to_decimal_fraction(bin_fraction) > upper_bound:
                bin_fraction = bin_fraction[:-1] + "0"

        encoded_img += bin_fraction

    return encoded_img


def encode_symbol(symbol, state, lower_bound, upper_bound):
    """
    Based on "Arithmetic Coding for Data Compression", Witten, Neal, and Cleary (1987)
    Accessed August 2021 from:
    https://www.researchgate.net/publication/200065260_Arithmetic_Coding_for_Data_Compression
    :param symbol:
    :return:
    """
    prob, orig_low, orig_high = state.intervals[symbol.item()]
    range = upper_bound - lower_bound
    high = lower_bound + range * orig_high
    low = lower_bound + range * orig_low

    return high, low


def decode_symbol(fraction, orig_low, orig_high, lower_bound, upper_bound):
    """
    :return:
    """
    range = upper_bound - lower_bound
    high = lower_bound + range * orig_high
    low = lower_bound + range * orig_low
    fraction = (fraction - low) / range  # Inverse operation from low = lower_bound + range * orig_low

    return high, low, fraction


def decode_image(encoding, state):
    n, m = state.shape
    num_of_blocks = (n * m) / BLOCK_SIZE ** 2
    # Decode each block separately using arithmetic coding

    bin_fractions = encoding.split(TERMINTAOR)
    new_m, new_n = m // BLOCK_SIZE, n // BLOCK_SIZE
    assert len(bin_fractions) == num_of_blocks

    for i in range(num_of_blocks):
        lower_bound = 0.0
        upper_bound = 1.0
        bin_fraction = "0." + bin_fractions[i]
        dec_fraction = convert_to_decimal_fraction(encoding)
        block_result = []

        while len(block_result) < BLOCK_SIZE ** 2:
            for orig_val in state.intervals.keys():
                # TODO: add sorted
                # The next encoded symbol has the point in its range
                prob, orig_low, orig_high = state.intervals[orig_val]
                if orig_low <= dec_fraction < orig_high:
                    block_result += [orig_val]
                    upper_bound, lower_bound, dec_fraction = decode_symbol(dec_fraction, orig_low, orig_high,
                                                                           lower_bound, upper_bound)
        block = np.array(block_result).reshape((BLOCK_SIZE, BLOCK_SIZE))

    decoded_img = np.array(block_result).reshape(state.shape).astype(np.int)
    return decoded_img


if __name__ == '__main__':
    a = np.array([1, 1, 2, 5, 5, 5, 2, 3]).reshape((2, 4))
    # a = cv2.imread('Mona-Lisa.bmp', cv2.IMREAD_GRAYSCALE)
    state = StateMachine(a)
    code = compress_image(a, state)
    decoded = decode_image(code, state)
    print(np.unique(a))
    print(np.unique(decoded))
    # plt.imshow(decoded)
    # plt.show()
