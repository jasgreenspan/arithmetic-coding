from utils import *
from golomb_encoding import encode_exp_golomb, decode_exp_golomb
from fractions import Fraction
import cv2
import matplotlib.pyplot as plt
from mpmath import findroot

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
            self.distribution = self._get_distribution_by_value(arg)

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
    lower_bound = 0
    upper_bound = 1

    # Encode each block separately using arithmetic coding
    for block in blocks:
        print("Encoding %d 'th block" % counter)
        counter += 1

        # Find sub-interval to encode block
        for val in np.nditer(block):
            # val_in_binary = np.binary_repr(val, width=BIT_LENGTH)
            # for bit in val_in_binary:
            #     upper_bound, lower_bound = encode_symbol(bit, state, lower_bound, upper_bound)
            upper_bound, lower_bound = encode_symbol(val, state, lower_bound, upper_bound)

    # Convert sub-interval to binary encoding
    # midway_point = (lower_bound + (upper_bound - lower_bound) / 2)
    # encoded_numerator = bin(midway_point.numerator)[2:]
    # encoded_denominator = bin(midway_point.denominator)[2:]
    # encoded_block = encode_exp_golomb(len(encoded_numerator), BOUNDS_LEN_GOLOMB_ORDER) \
    #                 + encode_exp_golomb(len(encoded_denominator), BOUNDS_LEN_GOLOMB_ORDER) \
    #                 + encoded_numerator + encoded_denominator
    # numerator = 6572871214251010991645015835723767699185332868727101738453459025591753131222773656239501961685759625879575572622025788670696762080518533126366051642782308460505728975981611782724123286687763333817166845386767722283836998994032035800013776435415878625375795258376736401811510841487574071528705463709989417998073509964642837664195228408259774581950163665659603178583295729637850463639970371094590994876792858123403030953539958298298364013596807217868641116175563873717337547952101140557313687013213389317626442926243865251968838572268147239130948362444699344663441903381540157463909046520501482626140160564433905650693928278127136266275068590622959285765945452439723934941356273629011380816466321279322636519678625587664876383934045540309004772731253577033775689971812183939295222857926983016993997254453195465130546619153513043343148236958574554946359710149305746866710522677595729174727061701810648564702895988389500653666851099934695310437371560426590114787494069845595672353426692165436356942790164140290755823985127088978186092911643131063503044839421994211369005034174058058757410478601866438380664254035726005410260420904652518854205058333051741709726650194520463255914066332310098115178303120314209331269107446838589305351466670934583244768838799315054073172209860580371819001638433887123170625532294577841083544280834500579605593245490957648218702631163031507630377055018842861694480635733160768485752959938160707885010213947550031762362564763394422206232240320429342602800176341146209732896899758098186908472428209258161200498026718291804893261950861622577479439197548003073392850465461871794476969418627446957523202836543579549954899474328584412192193778999721814928646839091493260806401024282507268353870177321260356024947714913358887713029313595588952636019138956482428287627182318505917693527179750329186729019891655535967550445624759121992448368882638623464068229377371396107982886202208689432993919625537801806799049708553043076076444951108738990408679676995504574379639680326443963989208525998944837767512996283248574766089371740535683979301222741938124364928006406145189219667611856078388242731592083780492829798901829153840671455078247448454702478079201359492273407193566634657898293431064082473461304998712791353840091602128763786368416500554991924346267156616133363600062741013965307187073896384670833548666070259645346875290956893659466069791641353117532501533462540163457032290929897876254971392682123771603035944877654920702120838890882174124875127726951139732569806377563699187420304738302111486578777504972965201057014304342271634463911672496950353178998224543681488199060783865571045165469917731749005134515478983386251845009608145759719654711971838780619892287255975183664377771984974123413746051630575374758414987162770309243459629559754690975603180426249
    # denominator = 6579455150369340738398768173698001731888089926963328369058724264605635358413999008930872993453113877985444222527700956241957713146605747937635305641143635305277795622928686551017092033219339895594898419444847118791400804080945102002785645851312571635981781249417506942869217810311122802506078489123797542891953379571963927778991391879335067772083719181237944925463884176068685380009695445133579681601941998367113682444923402547487888332460166168952779448270482245124591040679755110977781101038549005060110923084580129708331055961245047616491303595823963724749845085908491210850507198630755961885726635611140995395617876771863207548748962174947810896598745071400692205343887397123004153666874991199478132711350177251192629954521009955453841636539880117763810821127152963875200058950316008507835949264633968281557860450263737763309110204176412396692623163199540720269221998956935180311706151787281550483566175436536293850573331727144226510242332251934898625099284808420011210183039463234616378284294638513425454502266061123138168089852651460123463613494268206076535851455255331953614276941233936955440281224806866633172727904840298009614453973601228473519730053586543215226952214992442350352885569592368312499801864963183656043985698058657804603888939632641574585252959379104770792829326845232455547144161229526632684274378148783622088348802773996168558678319304359878280357652432197524423559995626470536126996743127087856426325211230986791316919733931497069989854387621815956421998507957942814992293910308013510330406970875308997749310017040819268993700287800577108134329747664360387251473308830268482106952707499584546656582320457272188317568323680233518658390312358861262781167151215877891955223684603705609985295602710030150071191372947404605285718142154466006777308952002642601557195800701167005881595118257033678762242905221349578432028897060212004026336057900949711351334832018346942557361752009908515002449239336329841740303902788631867230088916090694017676896239442160166400100069555501367090092717614333395424916342057811895732874978416678606695252866609383603110932554787581907429813128399699515329557149992647671336120722297763338734485197683230721848235402939061619730797020619053703819341959908407263835775702875502392170188303648101224458070846651934726888646283221714644576940721928870145720301695179343108686002504923386283540749233923929810069676540600715380764072131393633869637233111838682349185889004555969507122101595709658623360280262768856085073108707575022172142007785837552603503925492306794601378074554931372984881316269848979547644899426352348482720723761517749616810491190009427040453637625999531822388892025993437031693791151962068972725764878492938315224824212346057279131941207297621823031110854265664176385697560896910712258792081058682785823354519552

    encoded_block = ""
    current_bit = 0
    while True:
        # When to left of lower bound, make LSB '1'
        encoded_block += "1"
        code_as_fraction = Fraction(int(encoded_block, 2), 2 ** current_bit)
        # Check if binary fraction is in range, if so break
        if upper_bound > code_as_fraction >= lower_bound:
            break
        # When to right of upper bound, make LSB '0'
        if code_as_fraction > upper_bound:
            encoded_block = encoded_block[:-1] + "0"

        current_bit += 1

    encoded_img += encode_exp_golomb(len(encoded_block), BOUNDS_LEN_GOLOMB_ORDER) + encoded_block
    # encoded_img += encoded_block

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
    orig_low, orig_high = state.intervals[symbol.item()]
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
    # num_len, idx = decode_exp_golomb(encoding, BOUNDS_LEN_GOLOMB_ORDER, idx)
    # den_len, idx = decode_exp_golomb(encoding, BOUNDS_LEN_GOLOMB_ORDER, idx)
    # decoded_numerator, idx = decode_binary_string(encoding, idx, num_len)
    # decoded_denominator, idx = decode_binary_string(encoding, idx, den_len)
    # midway_point = Fraction(decoded_numerator, decoded_denominator)

    num_len, idx = decode_exp_golomb(encoding, BOUNDS_LEN_GOLOMB_ORDER, idx)
    decoded_numerator, idx = decode_binary_string(encoding, idx, num_len)
    midway_point = Fraction(decoded_numerator, 2 ** (num_len - 1))
    lower_bound = 0
    upper_bound = 1

    # Decode each block separately using arithmetic coding
    for i in range(num_of_blocks):
        print("Decoding %d 'th block" % i)
        # Decode the binary string representing the block
        # num_len, idx = decode_exp_golomb(encoding, BOUNDS_LEN_GOLOMB_ORDER, idx)
        # den_len, idx = decode_exp_golomb(encoding, BOUNDS_LEN_GOLOMB_ORDER, idx)
        # decoded_numerator, idx = decode_binary_string(encoding, idx, num_len)
        # decoded_denominator, idx = decode_binary_string(encoding, idx, den_len)

        # Reverse the arithmetic coding by going over all the ranges
        block_result = []

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
    a = cv2.imread('Mona-Lisa.bmp', cv2.IMREAD_GRAYSCALE)[:64, :64]
    # a = cv2.imread('Mona-Lisa.bmp', cv2.IMREAD_GRAYSCALE)
    state = StateMachine(a)
    code = compress_image(a, state)
    decoded = decompress_image(code)

    plt.imshow(a, cmap='gray')
    plt.show()
    plt.imshow(decoded, cmap='gray')
    plt.show()

    print((a == decoded).all())
