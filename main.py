from arithmetic_encoding import *
from golomb_encoding import *
import cv2


if __name__ == '__main__':
    a = cv2.imread('Mona-Lisa.bmp', cv2.IMREAD_GRAYSCALE)[:32, :32]

    # Calculate length of encoding image with Exp-Golomb Encoding
    golomb_encoding_len_by_pixel = np.array([exp_golomb_length(num, GOLOMB_ENC_ORDER) for num in np.nditer(a)])
    total_golomb_encoding_len = np.sum(golomb_encoding_len_by_pixel)

    # Encode image with Arithmetic Coding
    state = StateMachine(a)
    code = compress_image(a, state)
    total_arithmetic_coding_len = len(code)

    print("Encoded image using Order %d Exp-Golomb using %d bits" % (GOLOMB_ENC_ORDER, total_golomb_encoding_len))
    print("Encoded image using Arithmetic Coding using %d bits" % total_arithmetic_coding_len)

