#include <cmath>
#include <cstdint>
#include <cstring>
#include <omp.h>

extern "C" {

inline uint32_t read_u32(const uint8_t*& ptr, bool swap) {
    uint32_t val;
    std::memcpy(&val, ptr, 4);
    ptr += 4;
    return swap ? __builtin_bswap32(val) : val;
}

inline double read_f64(const uint8_t*& ptr, bool swap) {
    uint64_t val;
    std::memcpy(&val, ptr, 8);
    ptr += 8;
    if (swap) val = __builtin_bswap64(val);
    double d;
    std::memcpy(&d, &val, 8);
    return d;
}

inline double segment_length(double x1, double y1, double x2, double y2) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    return std::sqrt(dx * dx + dy * dy);
}

double length_linestring(const uint8_t*& ptr, bool swap) {
    uint32_t n = read_u32(ptr, swap);
    if (n < 2) {
        ptr += n * 16;
        return 0.0;
    }
    double x0 = read_f64(ptr, swap);
    double y0 = read_f64(ptr, swap);
    double total = 0.0;
    for (uint32_t i = 1; i < n; ++i) {
        double x1 = read_f64(ptr, swap);
        double y1 = read_f64(ptr, swap);
        total += segment_length(x0, y0, x1, y1);
        x0 = x1;
        y0 = y1;
    }
    return total;
}

double length_polygon(const uint8_t*& ptr, bool swap) {
    uint32_t rings = read_u32(ptr, swap);
    double total = 0.0;
    for (uint32_t i = 0; i < rings; ++i) {
        total += length_linestring(ptr, swap);
    }
    return total;
}

double length_wkb_internal(const uint8_t*& ptr, const uint8_t* end);

double length_collection(const uint8_t*& ptr, const uint8_t* end, bool swap, uint32_t parent_type) {
    uint32_t n = read_u32(ptr, swap);
    double total = 0.0;

    for (uint32_t i = 0; i < n; ++i) {
        const uint8_t* sub_ptr = ptr;
        if (sub_ptr + 5 > end) break;

        bool sub_swap = (*sub_ptr == 0); sub_ptr++;
        uint32_t sub_type = read_u32(sub_ptr, sub_swap) & 0xFF;

        ptr += 1 + 4; // advance original ptr past byte order + type

        switch (sub_type) {
            case 1:
                ptr += 16;
                break;
            case 2:
                total += length_linestring(ptr, sub_swap);
                break;
            case 3:
                total += length_polygon(ptr, sub_swap);
                break;
            case 4: case 5: case 6: case 7:
                total += length_collection(ptr, end, sub_swap, sub_type);
                break;
            default:
                break;
        }
    }

    return total;
}

double length_wkb_internal(const uint8_t*& ptr, const uint8_t* end) {
    if (ptr + 5 > end) return 0.0;
    bool swap = (*ptr == 0); ptr++;
    uint32_t type = read_u32(ptr, swap);
    uint32_t base_type = type & 0xFF;

    switch (base_type) {
        case 1:
            ptr += 16;
            return 0.0;
        case 2:
            return length_linestring(ptr, swap);
        case 3:
            return length_polygon(ptr, swap);
        case 4: case 5: case 6: case 7:
            return length_collection(ptr, end, swap, base_type);
        default:
            return 0.0;
    }
}

double compute_length_wkb(const uint8_t* data, size_t len) {
    const uint8_t* ptr = data;
    return length_wkb_internal(ptr, data + len);
}

double* fast_batch_length(char** wkb_ptrs, int* lens, int count) {
    double* out = new double[count];

    #pragma omp parallel for
    for (int i = 0; i < count; ++i) {
        const uint8_t* data = reinterpret_cast<const uint8_t*>(wkb_ptrs[i]);
        out[i] = compute_length_wkb(data, lens[i]);
    }

    return out;
}

void free_result_dbl(double* ptr) {
    delete[] ptr;
}

}

