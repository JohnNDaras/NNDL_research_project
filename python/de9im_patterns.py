import numpy as np


"""
H ΜΟΝΗ ΔΙΑΦΟΡΑ ΕΙΝΑΙ ΠΩΣ ΑΛΛΑΞΑ ΤΟ:

def decode_de9im_safe(u64):
    if u64 == 0:
        return 'FFFFFFFFF'  # disjoint fallback
    return ''.join(chr((u64 >> (8 * i)) & 0xFF) for i in range(8)) + 'F'


ΚΑΙ ΤΟ ΕΚΑΝΑ


def decode_de9im_safe(u64):
    # make sure we have a plain Python int
    u = int(u64)
    if u == 0:
        return 'FFFFFFFFF'  # disjoint fallback
    return ''.join(chr((u >> (8 * i)) & 0xFF) for i in range(8)) + 'F'


"""






########################################################################
# 1) Define a bitmask for each character in DE-9IM and each pattern char
########################################################################

# Which bit do we set for 'F','0','1','2'?
char_to_bit = {
    'F': 1,  # 0001 in binary
    '0': 2,  # 0010
    '1': 4,  # 0100
    '2': 8,  # 1000
}

# Which combined bits do we allow for a pattern char?
# e.g. 'T' means '0','1','2' => bits 2|4|8 = 14

pattern_to_bits = {
    'F':  char_to_bit['F'],                 # must be 'F'
    '0':  char_to_bit['0'],                 # must be '0'
    '1':  char_to_bit['1'],                 # must be '1'
    '2':  char_to_bit['2'],                 # must be '2'
    'T':  char_to_bit['0']|char_to_bit['1']|char_to_bit['2'],  # any of '0','1','2'
    '*':  char_to_bit['F']|char_to_bit['0']|char_to_bit['1']|char_to_bit['2'],  # anything
}



pattern_to_bits_array = np.zeros(128, dtype=np.uint8)
for ch, val in pattern_to_bits.items():
    pattern_to_bits_array[ord(ch)] = val

#######################################################################
# 2) Convert an array of DE-9IM strings to a NumPy array of shape (N,9)
#    where each entry is the integer bitmask for that character
#######################################################################


def encode_de9im_strings(de9im_array):
    """
    Fully vectorized function to convert an array of DE-9IM strings into a (N,9) uint8 bitmask array.
    Assumes all strings are of length 9 and contain only valid characters ('F', '0', '1', '2').
    """
    # Convert the char_to_bit mapping into a lookup table (ASCII-based)
    char_to_bit_array = np.zeros(128, dtype=np.uint8)  # covers standard ASCII
    for char, bit in char_to_bit.items():
        char_to_bit_array[ord(char)] = bit
    # Step 1: Convert input to uppercase byte strings of shape (N,)
    de9im_array = np.char.upper(np.asarray(de9im_array, dtype='U9'))

    # Step 2: View each character in each string as a separate column → shape (N, 9)
    # This avoids any for-loop or list comprehension
    char_matrix = de9im_array.view('U1').reshape(-1, 9)

    # Step 3: Convert to ASCII integer codes → shape (N, 9)
    ascii_matrix = np.vectorize(ord)(char_matrix)

    # Step 4: Use lookup table to get corresponding bitmask for each character
    bitmask_matrix = char_to_bit_array[ascii_matrix]

    return bitmask_matrix


#######################################################################
# 3) Define classes that store the pattern in a bitmask array of shape (9,)
#    then use purely NumPy logic for matches_array
#######################################################################

class Pattern:
    def __init__(self, pattern_string):
        pattern_string = pattern_string.upper()
        if len(pattern_string) != 9:
            raise ValueError("DE-9IM pattern must be exactly 9 chars long")

        # Convert string to NumPy array of characters
        chars = np.array(list(pattern_string), dtype='U1')

        # Convert characters to ASCII codes
        ascii_codes = np.vectorize(ord)(chars)

        # Use lookup table to get pattern bitmask values
        self._pattern_bits = pattern_to_bits_array[ascii_codes]

    def matches_array(self, de9im_encoded):
        """
        de9im_encoded: shape (N,9), each entry is the bitmask
        Return a boolean array of shape (N,) telling which rows match.
        """
        # We want to ensure for each position j, (de9im_encoded[row,j] & self._pattern_bits[j]) != 0
        # Then we do np.all(..., axis=1).
        # We'll broadcast self._pattern_bits to shape (1,9), so we do elementwise &.
        mask_2d = (de9im_encoded & self._pattern_bits) != 0  # shape (N,9) -> bool
        return np.all(mask_2d, axis=1)  # shape (N,)


class AntiPattern:
    def __init__(self, anti_pattern_string):
        anti_pattern_string = anti_pattern_string.upper()
        if len(anti_pattern_string) != 9:
            raise ValueError("DE-9IM pattern must be exactly 9 chars long")

        # Convert string to NumPy array of characters
        chars = np.array(list(anti_pattern_string), dtype='U1')

        # Convert characters to ASCII codes
        ascii_codes = np.vectorize(ord)(chars)

        # Use lookup table to get pattern bitmask values
        self._pattern_bits = pattern_to_bits_array[ascii_codes]


    def matches_array(self, de9im_encoded):
        """
        The old logic was:
           AntiPattern('FF*FF****').matches(s) => not all(m in DIMS[p])
        So if a row WOULD match the pattern, we invert it.
        """
        mask_2d = (de9im_encoded & self._pattern_bits) != 0
        # "would_match" means all 9 positions are satisfied
        would_match = np.all(mask_2d, axis=1)
        # AntiPattern says "return True if it does NOT match"
        return ~would_match

class NOrPattern:
    def __init__(self, pattern_strings):
        num_patterns = len(pattern_strings)

        # Validate all patterns and convert to uppercase
        if not all(isinstance(p, str) and len(p) == 9 for p in pattern_strings):
            raise ValueError("All DE-9IM patterns must be 9-character strings")

        # Step 1: Convert to uppercase using NumPy
        pattern_array = np.char.upper(np.array(pattern_strings, dtype='U9'))  # shape (N,)

        # Step 2: Convert to (N, 9) character matrix (U1)
        char_matrix = pattern_array.view('U1').reshape(-1, 9)

        # Step 3: Convert to ASCII codes (vectorized)
        ascii_matrix = np.vectorize(ord)(char_matrix)  # shape (N, 9)

        # Step 4: Convert to bitmask matrix using lookup
        self._all_patterns_bits = pattern_to_bits_array[ascii_matrix]  # shape (N, 9)

    def matches_array(self, de9im_encoded):
        """
        Vectorized match across multiple patterns (OR logic).
        de9im_encoded: shape (N, 9), dtype=uint8
        self._all_patterns_bits: shape (P, 9), dtype=uint8
        Returns:
            Boolean mask of shape (N,), True where any pattern matched.
        """
        # Expand dimensions to broadcast: (N, 1, 9) & (1, P, 9) => (N, P, 9)
        # Result is True where each de9im position matches the pattern's allowed bits
        match_matrix = (de9im_encoded[:, None, :] & self._all_patterns_bits[None, :, :]) != 0  # shape (N, P, 9)

        # For each (N, P), check if all 9 positions match
        full_match = np.all(match_matrix, axis=2)  # shape (N, P)

        # Now check if any pattern matched for each geometry pair
        any_match = np.any(full_match, axis=1)  # shape (N,)

        return any_match
#######################################################################
# 4) Finally, define your familiar patterns using these classes
#######################################################################

contains = Pattern('T*****FF*')
crosses_lines = Pattern('0********')
crosses_1 = Pattern('T*T******')
crosses_2 = Pattern('T*****T**')
disjoint = AntiPattern('FF*FF****')
equal = Pattern('T*F**FFF*')
intersects_de9im = AntiPattern('FF*FF****')
overlaps1 = Pattern('T*T***T**')
overlaps2 = Pattern('1*T***T**')
touches = NOrPattern(['FT*******', 'F**T*****', 'F***T****'])
within = Pattern('T*F**F***')
covered_by = NOrPattern(['T*F**F***','*TF**F***','**FT*F***','**F*TF***'])
covers = NOrPattern(['T*****FF*','*T****FF*','***T**FF*','****T*FF*'])

