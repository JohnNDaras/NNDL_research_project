import numpy as np
import ctypes, os
from ctypes import POINTER, c_int, c_float

BASE_LIB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib'))

class CandidateStats:
    def __init__(self, lib_path=None):
        if lib_path is None:
            lib_path = os.path.join(BASE_LIB, 'libcandidate_stats.so')

        self._lib = ctypes.CDLL(lib_path)

        self._lib.compute_candidate_stats.argtypes = [
            POINTER(c_int), c_int,              # target_ids, N
            POINTER(c_int), POINTER(c_int),     # candidate_offsets, candidate_values
            POINTER(c_int),                     # frequency
            POINTER(c_float), POINTER(c_int)    # out_freq_sums, out_candidate_counts
        ]
        self._lib.compute_candidate_stats.restype = None

    def compute(self, target_ids, candidate_offsets, candidate_values, frequency):
        N = len(target_ids)
        target_ids = np.asarray(target_ids, dtype=np.int32)
        candidate_offsets = np.asarray(candidate_offsets, dtype=np.int32)
        candidate_values = np.asarray(candidate_values, dtype=np.int32)
        frequency = np.asarray(frequency, dtype=np.int32)

        out_freq_sums = np.zeros(N, dtype=np.float32)
        out_counts = np.zeros(N, dtype=np.int32)

        self._lib.compute_candidate_stats(
            target_ids.ctypes.data_as(POINTER(c_int)), N,
            candidate_offsets.ctypes.data_as(POINTER(c_int)),
            candidate_values.ctypes.data_as(POINTER(c_int)),
            frequency.ctypes.data_as(POINTER(c_int)),
            out_freq_sums.ctypes.data_as(POINTER(c_float)),
            out_counts.ctypes.data_as(POINTER(c_int))
        )

        return out_freq_sums, out_counts

