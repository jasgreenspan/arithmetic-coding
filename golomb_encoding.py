from utils import *

def encode_golomb_rice(num, order):
    """
    Does golomb encoding
    :param num: number to encode
    :param order: a divisor to calculate the quotient and remainder
    :return: encoded num
    """
    quotient = int(np.floor(num/order))
    remainder = num % order
    quotient_encoding = "1" * quotient + "0"
    b = np.ceil(np.log2(order)).astype('int')
    k = max((2 ** b) - order, 0)
    if remainder < k:
        # code r in binary using b bits
        remainder_encoding = ("{0:0" + str(b-1) + "b}").format(remainder)

    else:
        # code (r+k) in binary using b+1 bits
        remainder_encoding = ("{0:0" + str(b) + "b}").format(remainder+k)
    return quotient_encoding + remainder_encoding


def decode_golomb_rice(binary_str, order, start_i):
    """
    Decodes binary string encoded with golomb rice
    :param binary_str:
    :param order:
    :param start_i:
    :return:
    """
    i = start_i
    b = np.ceil(np.log2(order)).astype('int')
    k = max((2 ** b) - order, 0)

    # Let s ← the number of consecutive ones in the input (we stop when we read a 0).
    quotient = binary_str.count('1', i, binary_str[i:].find('0') + i)
    i += quotient + 1  # Skip over 0 separating bit

    # Let x ← the next b−1 bits in the input.
    remainder_str = binary_str[i: i + b - 1]
    i += max(b - 1, 0)
    remainder = 0 if remainder_str == '' else int(remainder_str, 2)

    # Calculate the number with quotient and remainder
    if remainder < k:
        num = quotient * order + remainder
    else:
        last_bit = binary_str[i: i + 1]
        i += 1
        if last_bit == '':
            num = quotient * order + remainder * 2 - k
        else:
            remainder = remainder * 2 + int(last_bit, 2)
            num = quotient * order + remainder - k

    return num, i


def convert_neg_num(num):
    """
    Converts negative numbers as follows 0 -> 1, 1 -> 2, -1 -> 3,...
    :param num:
    :return:
    """
    return abs((2 * num) - 1) if num <= 0 else abs(2 * num)


def revert_neg_num(num):
    """
    Returns mapped negative numbers to their original
    :param num:
    :return:
    """
    result = (num + 1) / 2 if (num % 2) == 0 else (num / 2) * -1
    return int(result)


def encode_exp_golomb(num, order):
    """
    Encode exp golomb
    :param num: the number to encode
    :param order: the order k
    :return: encoding string
    """
    converted_num = convert_neg_num(num)
    quotient = bin(abs(converted_num) // (2 ** order) + 1)[2:]
    quotient = '0' * (len(quotient) - 1) + quotient
    remainder = '' if order == 0 else ("{0:0" + str(order) + "b}").format(converted_num % 2 ** order)

    return quotient + remainder


def decode_exp_golomb(binary_str, order, start_i):
    """
    Decode a binary exp Golomb code
    :param order: the order of the encoding
    :param binary_str: the encoded string
    :return: the value
    """
    i = start_i

    num_of_zeros = binary_str.count('0', i, binary_str[i:].find('1') + i)
    quotient = int(binary_str[i + num_of_zeros: i + (2 * num_of_zeros) + 1], 2)
    i += (2 * num_of_zeros) + 1

    # If k == 0, there is no remainder bit
    if order > 0:
        remainder = int(binary_str[i: i + order], 2)
        i += order
    else:
        remainder = 0

    num = (((quotient - 1) * (2 ** order)) + remainder)
    reverted_num = revert_neg_num(num)
    return reverted_num, i


def exp_golomb_length(num, order):
    """
    Calculate the length of encoding a number with exp golomb
    :param num: the number
    :param order: the order of the encoding
    :return:
    """
    converted_num = convert_neg_num(num)
    quotient = np.floor(converted_num / 2 ** order)

    # the quotient is sent using 2 * floor(log2(y + 1)) + 1 bits and the remainder using k bits
    quotient_len = 2 * np.floor(np.log2(quotient + 1)) + 1

    return quotient_len + order