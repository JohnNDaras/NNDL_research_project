import ctypes
import os
from pathlib import Path
from ctypes import POINTER, c_char_p, c_int

# figure out absolute path to our project’s lib/ dir
BASE_LIB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib'))

class CsvReader:
    def __init__(self, lib_path: str = None):
        if lib_path is None:
            lib_path = os.path.join(BASE_LIB, 'libwkt.so')
        self.lib = ctypes.CDLL(lib_path)

        # Function signatures
        self.lib.read_wkb_for_python.argtypes = [
            ctypes.c_char_p,                      # filepath
            ctypes.c_char,                        # delimiter
            ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_char))),  # ***out_results
            ctypes.POINTER(ctypes.c_int),         # *out_count
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int))  # **out_lengths
        ]
        self.lib.read_wkb_for_python.restype = None

        self.lib.free_wkb_for_python.argtypes = [
            ctypes.POINTER(ctypes.POINTER(ctypes.c_char)),  # **results
            ctypes.c_int,                                   # count
            ctypes.POINTER(ctypes.c_int)                    # *lengths
        ]
        self.lib.free_wkb_for_python.restype = None

        # Internal storage
        self.sourceData = []
        self.targetData = []

    def readAllEntities(self, filepath: str, delimiter='\t'):
        # Convert inputs
        filepath_bytes = Path(filepath).expanduser().resolve().as_posix().encode()
        delim_byte = delimiter.encode()[0]

        # Prepare C variables
        results = ctypes.POINTER(ctypes.POINTER(ctypes.c_char))()
        lengths = ctypes.POINTER(ctypes.c_int)()
        count = ctypes.c_int()

        # Call C++ shared library function
        self.lib.read_wkb_for_python(
            filepath_bytes, delim_byte,
            ctypes.byref(results), ctypes.byref(count), ctypes.byref(lengths)
        )

        # Convert to Python list of WKB bytes
        wkb_list = [
            ctypes.string_at(results[i], lengths[i]) for i in range(count.value)
        ]

        # Free C++ memory
        self.lib.free_wkb_for_python(results, count, lengths)

        return wkb_list



