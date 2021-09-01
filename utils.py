from functools import reduce
from math import sqrt

import numpy as np
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt

BLOCK_SIZE = 8


############################################################
# Image Processing Utils
############################################################
def block_img(orig_img):
    """
    Make image into array of 8 * 8 blocks
    :param orig_img: original image, of shape (m, n)
    :return: the blocked image, of shape (m / 8, n / 8, 8, 8)
    """
    m, n = orig_img.shape
    new_m, new_n = m // BLOCK_SIZE, n // BLOCK_SIZE
    return orig_img.reshape((new_m, BLOCK_SIZE, new_n, BLOCK_SIZE)).swapaxes(1, 2)


def deblock_img(blocked_img):
    """
    Make array of 8 * 8 blocks into a regular image
    :param blocked_img: the blocked image, of shape (m / 8, n / 8, 8, 8)
    :return: new image, of shape (m, n)
    """
    block_m, block_n, _, _ = blocked_img.shape
    return blocked_img.swapaxes(1, 2).reshape(block_m * BLOCK_SIZE, block_n * BLOCK_SIZE)


def scaling(x):
    """
    Scaling from [0, 255] to [−128, 128) and back.
    :return:
    """
    normalized = x.astype('int16') - 128
    return normalized


def unscale(x):
    unscaled = x.astype('int16') + 128
    unscaled = np.clip(unscaled, 0, 255)
    return unscaled


def dct2(a):
    """
    implement 2D DCT
    :param a:
    :return:
    """
    return dct(dct(a.T, norm='ortho').T, norm='ortho')


def idct2(a):
    """
    implement 2D IDCT
    :param a:
    :return:
    """
    return idct(idct(a.T, norm='ortho').T, norm='ortho')


############################################################
# Image Compression Utils
############################################################
def zigzag(x):
    """
    Translate between N × N matrix to N^2 vector through zigzag method
    :return:
    """
    n = x.shape[0]

    # flip x to be in minor diagonal
    flip_x = np.flipud(x)
    diagonals = []
    for k in range(1 - n, n):
        # get the anti-diagonals
        diag = np.diagonal(flip_x, k)[::(2 * (k % 2) - 1)].flatten()
        diagonals += [diag]

    return np.concatenate(diagonals)


def zig_zag_index(i, j, n):
    k = i + j
    # for bottom right of matrix
    if k >= n:
        return (n ** 2) - 1 - zig_zag_index(n - 1 - i, n - 1 - j, n)

    # for top left of matrix
    # scan at position [k, 0] is preceded by 1+2+…+k = k*(k+1)/2 items
    num_items = k * (k + 1) // 2
    return num_items + i if k % 2 != 0 else num_items + j


def inv_zigzag(inv_x):
    """
    Invert translate between N^2 vector to N × N matrix through zigzag method
    :return:
    """
    n = int(round(np.sqrt(inv_x.size)))
    x = np.empty((n, n), dtype='int64')
    for i in range(n):
        for j in range(n):
            idx = zig_zag_index(i, j, n)
            x[i, j] = inv_x[idx]
    return x


############################################################
# Arithmetic Coding Utils
############################################################
def convert_to_binary_fraction(dec_fraction):
    """
    Convert a decimal fraction in range [0.0, 1) to a binary fraction
    For further reading: https://www.electronics-tutorials.ws/binary/binary-fractions.html
    :param dec_fraction: the decimal fraction
    :return: the binary fraction
    """
    bin_fraction = '0.'
    frac = None

    while frac != '0':
        carry_bit, frac = str(f'{dec_fraction * 2:.20f}').split(".", 1)
        dec_fraction = float('0.' + frac)
        bin_fraction += str(carry_bit)

    return bin_fraction


def convert_to_decimal_fraction(bin_fraction):
    """
    Convert a binary fraction to a decimal fraction in range [0.0, 1)
    Using the formula 0.b1b2b3... = b1 * 2^(-1) + b2 * 2^(-2) + ...
    :param bin_fraction: the binary fraction
    :return: the decimal fraction
    """
    frac = bin_fraction.split(".")[1]

    dec_fraction = 0.0
    for i in range(len(frac)):
        dec_fraction += (int(frac[i]) / 2 ** (i + 1))

    return dec_fraction


############################################################
# Math Utils
############################################################
def mse(img1, img2):
    """
    compute mse of two images
    :param img1:
    :param img2:
    :return:
    """
    return np.mean((img1 - img2) ** 2)


def psnr(img1, img2):
    """
    compute psnr of two images
    :param img1:
    :param img2:
    :return:
    """
    error = mse(img1, img2)
    max_pixel = 255.0
    ratio = 10 * np.log10((max_pixel ** 2) / error)
    return ratio

def factors(n):
    """
    From stackoverflow user Steinar Lima
    Accessed August 2021:
    https://stackoverflow.com/questions/6800193/what-is-the-most-efficient-way-of-finding-all-the-factors-of-a-number-in-python
    :param n:
    :return:
    """
    step = 2 if n % 2 else 1
    return set(reduce(list.__add__,
                      ([i, n // i] for i in range(1, int(sqrt(n)) + 1, step) if n % i == 0)))

############################################################
# Graph Utils
############################################################
def make_rate_graph(ratio_points, compare_points, quality_factors, y_label, textstr, loc, graph_title,
                    additional_line=0):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if additional_line != 0:
        plt.axvline(x=additional_line, label='Zero Encoding Avg', c='m')

    plt.plot(ratio_points, compare_points, '-ro')
    for i, txt in enumerate(quality_factors):
        ax.annotate(txt, (ratio_points[i], compare_points[i]), verticalalignment='top',
                    fontsize=10)

    plt.xlabel('bpp (bits per pixel)')
    plt.ylabel(y_label)

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in left in axes coords
    ax.text(0.05, loc, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    plt.legend()
    plt.title(graph_title)
    plt.savefig(graph_title)
    plt.show()


def make_joint_rate_graph(ratios, compares, quality_factors, lines, y_label, graph_title, text_str):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for ratio_points, compare_points, line in zip(ratios, compares, lines):
        plt.plot(ratio_points, compare_points, '-o', label=line)

    for i, txt in enumerate(quality_factors):
        for ratio_points, compare_points in zip(ratios, compares):
            ax.annotate(txt, (ratio_points[i], compare_points[i]), verticalalignment='top', fontsize=10)

    plt.xlabel('bpp (bits per pixel)')
    plt.ylabel(y_label)

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in left in axes coords
    ax.text(0.85, 0.5, text_str, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    plt.title(graph_title)
    plt.legend()
    plt.savefig(graph_title)
    plt.show()
