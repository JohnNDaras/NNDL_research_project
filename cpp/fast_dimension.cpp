#include <cstdint>
#include <cstring>
#include <omp.h>
#include <algorithm>

extern "C" {

inline uint32_t read_u32(const uint8_t*& ptr, bool swap) {
    uint32_t val;
    std::memcpy(&val, ptr, 4);
    ptr += 4;
    return swap ? __builtin_bswap32(val) : val;
}

int dimension_from_type(uint32_t type_id) {
    uint32_t base = type_id & 0xFF;
    switch (base) {
        case 1: case 4:  return 0; // Point, MultiPoint
        case 2: case 5:  return 1; // LineString, MultiLineString
        case 3: case 6:  return 2; // Polygon, MultiPolygon
        default:         return -1;
    }
}

int dimension_wkb_internal(const uint8_t*& ptr, const uint8_t* end);

int dimension_collection(const uint8_t*& ptr, const uint8_t* end, bool swap) {
    if (ptr + 4 > end) return -1;
    uint32_t n = read_u32(ptr, swap);
    int max_dim = -1;
    for (uint32_t i = 0; i < n; ++i) {
        int dim = dimension_wkb_internal(ptr, end);
        max_dim = std::max(max_dim, dim);
    }
    return max_dim;
}

int dimension_wkb_internal(const uint8_t*& ptr, const uint8_t* end) {
    if (ptr + 5 > end) return -1;
    bool swap = (*ptr == 0); ptr++;
    uint32_t type = read_u32(ptr, swap);
    uint32_t base_type = type & 0xFF;

    if (base_type == 7) { // GeometryCollection
        return dimension_collection(ptr, end, swap);
    }
    return dimension_from_type(type);
}

int compute_dimension(const uint8_t* data, size_t len) {
    const uint8_t* ptr = data;
    const uint8_t* end = data + len;
    return dimension_wkb_internal(ptr, end);
}

int* fast_batch_dimensions(char** wkb_ptrs, int* lens, int count) {
    int* out = new int[count];
    #pragma omp parallel for
    for (int i = 0; i < count; ++i) {
        const uint8_t* data = reinterpret_cast<const uint8_t*>(wkb_ptrs[i]);
        out[i] = compute_dimension(data, lens[i]);
    }
    return out;
}

void free_result_int(int* ptr) {
    delete[] ptr;
}

}

