from utils import *
import cv2
import matplotlib.pyplot as plt


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
    lower_bound = 0.0
    upper_bound = 1.0
    subinterval = 1.0

    # Find subinterval to encode image
    for val in np.nditer(img):
        prob, orig_start, orig_end = state.intervals[val.item()]
        start = lower_bound + subinterval * orig_start
        end = lower_bound + subinterval * orig_end
        subinterval = start - end

        lower_bound = start
        upper_bound = end

    # Convert subinterval to binary encoding
    bin_fraction = "0."
    while True:
        # When to left of lower bound, make LSB '1'
        bin_fraction += "1"
        if convert_to_decimal_fraction(bin_fraction) > lower_bound:
            break
        # When to right of upper bound, make LSB '0'
        if convert_to_decimal_fraction(bin_fraction) > upper_bound:
            bin_fraction = bin_fraction[:-1] + "0"

    return bin_fraction


def encode_symbol(symbol, cum_freq):
    """
    Based on "Arithmetic Coding for Data Compression", Witten, Neal, and Cleary (1987)
    Accessed August 2021 from:
    https://www.researchgate.net/publication/200065260_Arithmetic_Coding_for_Data_Compression
    :param symbol:
    :param cum_freq:
    :return:
    """
    range = high - low
    high = low + range * cum_freq[symbol - 1]
    low = low + range * cum_freq[symbol]

def decode_symbol(cum_freq):
    """

    :param cum_freq:
    :return:
    """
    range = low - high
    high = low + range * cum_freq[symbol - 1]
    low = low + range * cum_freq[symbol]

    return symbol

def decompress_image(encoding, state):
    dec_fraction = convert_to_decimal_fraction(encoding)
    n, m = state.shape
    result = []
    while len(result) < n * m:
        for orig_val in state.intervals.keys():
            # TODO: add sorted
            prob, orig_start, orig_end = state.intervals[orig_val]
            if orig_start <= dec_fraction < orig_end:
                result += [orig_val]
                #
                # start = orig_start
                # end = orig_end
                # subinterval = end - start
                # dec_fraction = (dec_fraction - start) / subinterval
                start = lower_bound + subinterval * orig_start
                end = lower_bound + subinterval * orig_end
                subinterval = start - end

                lower_bound = start
                upper_bound = end

    decoded_img = np.array(result).reshape(state.shape).astype(np.int)
    return decoded_img


if __name__ == '__main__':
    a = np.array([1, 1, 2, 5, 5, 5, 2, 3]).reshape((2, 4))
    # a = cv2.imread('Mona-Lisa.bmp', cv2.IMREAD_GRAYSCALE)
    state = StateMachine(a)
    code = compress_image(a, state)
    decoded = decompress_image(code, state)
    print(np.unique(a))
    print(np.unique(decoded))
    # plt.imshow(decoded)
    # plt.show()
