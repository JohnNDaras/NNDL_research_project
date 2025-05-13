import numpy as np
import ctypes, os
from ctypes import (
    POINTER, c_char_p, c_int, c_uint32, c_double, c_uint64, c_float
)
#from shapely.wkb import dumps as wkb_dumps

BASE_LIB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib'))

# Structure to hold bbox pair results
class IntPair(ctypes.Structure):
    _fields_ = [("source_id", c_int), ("target_id", c_int)]

class FastGeom:
    def __init__(self,
                 coord_lib_path=None,
                 length_lib_path=None,
                 bounds_lib_path=None,
                 relate_lib_path=None,
                 dimension_lib_path=None,
                 gridbbox_lib_path=None):
        if coord_lib_path is None:
            coord_lib_path = os.path.join(BASE_LIB, 'libfast_count_coords.so')
        if length_lib_path is None:
            length_lib_path = os.path.join(BASE_LIB, 'libfast_length.so')
        if bounds_lib_path is None:
            bounds_lib_path = os.path.join(BASE_LIB, 'libfast_bounds.so')
        if relate_lib_path is None:
            relate_lib_path = os.path.join(BASE_LIB, 'librelate_wkb_u64.so')
        if dimension_lib_path is None:
            dimension_lib_path = os.path.join(BASE_LIB, 'libfast_dimension.so')
        if gridbbox_lib_path is None:
            gridbbox_lib_path = os.path.join(BASE_LIB, 'libgrid_bbox_join.so')


        # Coordinate count
        self._coord_lib = ctypes.CDLL(coord_lib_path)
        self._coord_lib.fast_batch_count_coords.argtypes = [POINTER(c_char_p), POINTER(c_int), c_int]
        self._coord_lib.fast_batch_count_coords.restype = POINTER(c_uint32)
        self._coord_lib.free_result_u32.argtypes = [POINTER(c_uint32)]

        # Length
        self._length_lib = ctypes.CDLL(length_lib_path)
        self._length_lib.fast_batch_length.argtypes = [POINTER(c_char_p), POINTER(c_int), c_int]
        self._length_lib.fast_batch_length.restype = POINTER(c_double)
        self._length_lib.free_result_dbl.argtypes = [POINTER(c_double)]

        # Bounds
        self._bounds_lib = ctypes.CDLL(bounds_lib_path)
        self._bounds_lib.fast_batch_bounds.argtypes = [POINTER(c_char_p), POINTER(c_int), c_int]
        self._bounds_lib.fast_batch_bounds.restype = POINTER(c_double)
        self._bounds_lib.free_result_dbl.argtypes = [POINTER(c_double)]

        # Relate (DE-9IM)
        self._relate_lib = ctypes.CDLL(relate_lib_path)
        self._relate_lib.relate_batch_wkb_u64.argtypes = [
            POINTER(c_char_p), POINTER(c_int),
            POINTER(c_char_p), POINTER(c_int),
            c_int
        ]
        self._relate_lib.relate_batch_wkb_u64.restype = POINTER(c_uint64)
        self._relate_lib.free_result_u64.argtypes = [POINTER(c_uint64)]

        # Dimension
        self._dimension_lib = ctypes.CDLL(dimension_lib_path)
        self._dimension_lib.fast_batch_dimensions.argtypes = [POINTER(c_char_p), POINTER(c_int), c_int]
        self._dimension_lib.fast_batch_dimensions.restype = POINTER(c_int)
        self._dimension_lib.free_result_int.argtypes = [POINTER(c_int)]

        # Grid-based bbox intersection
        self._grid_lib = ctypes.CDLL(gridbbox_lib_path)
        self._grid_lib.grid_bbox_join.argtypes = [
            POINTER(c_float), c_int,
            POINTER(c_float), c_int,
            c_float, c_float, c_float, c_float,
            c_int, c_int,
            POINTER(c_int)
        ]
        self._grid_lib.grid_bbox_join.restype = POINTER(IntPair)
        self._grid_lib.free_grid_pairs.argtypes = [POINTER(IntPair)]
        self._grid_lib.free_grid_pairs.restype = None

    def get_num_of_points(self, wkb_list):
        N = len(wkb_list)
        lens = np.array([len(wkb) for wkb in wkb_list], dtype=np.int32)
        ptrs = (c_char_p * N)(*wkb_list)
        result_ptr = self._coord_lib.fast_batch_count_coords(ptrs, lens.ctypes.data_as(POINTER(c_int)), N)
        result = np.ctypeslib.as_array(result_ptr, shape=(N,)).copy()
        self._coord_lib.free_result_u32(result_ptr)
        return result

    def get_lengths(self, wkb_list):
        N = len(wkb_list)
        lens = np.array([len(wkb) for wkb in wkb_list], dtype=np.int32)
        ptrs = (c_char_p * N)(*wkb_list)
        result_ptr = self._length_lib.fast_batch_length(ptrs, lens.ctypes.data_as(POINTER(c_int)), N)
        result = np.ctypeslib.as_array(result_ptr, shape=(N,)).copy()
        self._length_lib.free_result_dbl(result_ptr)
        return result

    def get_bounds(self, wkb_list):
        N = len(wkb_list)
        lens = np.array([len(wkb) for wkb in wkb_list], dtype=np.int32)
        ptrs = (c_char_p * N)(*wkb_list)
        result_ptr = self._bounds_lib.fast_batch_bounds(ptrs, lens.ctypes.data_as(POINTER(c_int)), N)
        result = np.ctypeslib.as_array(result_ptr, shape=(N * 4,)).reshape(N, 4).copy()
        self._bounds_lib.free_result_dbl(result_ptr)
        return result

    def get_dimensions(self, wkb_list):
        N = len(wkb_list)
        lens = np.array([len(wkb) for wkb in wkb_list], dtype=np.int32)
        ptrs = (c_char_p * N)(*wkb_list)
        result_ptr = self._dimension_lib.fast_batch_dimensions(ptrs, lens.ctypes.data_as(POINTER(c_int)), N)
        result = np.ctypeslib.as_array(result_ptr, shape=(N,)).copy()
        self._dimension_lib.free_result_int(result_ptr)
        return result

    def relate(self, wkb_list_1, wkb_list_2):
        assert len(wkb_list_1) == len(wkb_list_2), "Input lists must be the same length"
        N = len(wkb_list_1)
        lens1 = np.array([len(wkb) for wkb in wkb_list_1], dtype=np.int32)
        lens2 = np.array([len(wkb) for wkb in wkb_list_2], dtype=np.int32)
        ptrs1 = (c_char_p * N)(*wkb_list_1)
        ptrs2 = (c_char_p * N)(*wkb_list_2)
        result_ptr = self._relate_lib.relate_batch_wkb_u64(
            ptrs1, lens1.ctypes.data_as(POINTER(c_int)),
            ptrs2, lens2.ctypes.data_as(POINTER(c_int)),
            N
        )
        result = np.ctypeslib.as_array(result_ptr, shape=(N,)).copy()
        self._relate_lib.free_result_u64(result_ptr)
        return result

    def grid_bbox_intersect(self, source_bounds, target_bounds, extent, grid_x=64, grid_y=64):
        """
        Perform fast spatial join using a grid index on bounding boxes.
        """
        n_src = source_bounds.shape[0]
        n_tgt = target_bounds.shape[0]
        src_flat = source_bounds.astype(np.float32).ravel()
        tgt_flat = target_bounds.astype(np.float32).ravel()
        minx, miny, maxx, maxy = map(float, extent)

        out_count = c_int()
        result_ptr = self._grid_lib.grid_bbox_join(
            src_flat.ctypes.data_as(POINTER(c_float)), n_src,
            tgt_flat.ctypes.data_as(POINTER(c_float)), n_tgt,
            minx, miny, maxx, maxy,
            grid_x, grid_y,
            ctypes.byref(out_count)
        )

        raw = np.ctypeslib.as_array(result_ptr, shape=(out_count.value,))
        structured = np.frombuffer(raw.data, dtype=[("source_id", np.int32), ("target_id", np.int32)])
        pairs = np.column_stack((structured["source_id"], structured["target_id"]))
        self._grid_lib.free_grid_pairs(result_ptr)
        return pairs

