#include <cstdint>
#include <cstring>
#include <omp.h>
#include <limits>

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

void update_bounds(double& minx, double& miny, double& maxx, double& maxy, double x, double y) {
    if (x < minx) minx = x;
    if (y < miny) miny = y;
    if (x > maxx) maxx = x;
    if (y > maxy) maxy = y;
}

void scan_points(const uint8_t*& ptr, uint32_t count, bool swap,
                 double& minx, double& miny, double& maxx, double& maxy) {
    for (uint32_t i = 0; i < count; ++i) {
        double x = read_f64(ptr, swap);
        double y = read_f64(ptr, swap);
        update_bounds(minx, miny, maxx, maxy, x, y);
    }
}

void bounds_wkb_internal(const uint8_t*& ptr, const uint8_t* end,
                         double& minx, double& miny, double& maxx, double& maxy);

void bounds_collection(const uint8_t*& ptr, const uint8_t* end, bool swap,
                       double& minx, double& miny, double& maxx, double& maxy) {
    uint32_t n = read_u32(ptr, swap);
    for (uint32_t i = 0; i < n; ++i) {
        bounds_wkb_internal(ptr, end, minx, miny, maxx, maxy);
    }
}

void bounds_wkb_internal(const uint8_t*& ptr, const uint8_t* end,
                         double& minx, double& miny, double& maxx, double& maxy) {
    if (ptr + 5 > end) return;

    bool swap = (*ptr == 0); ptr++;
    uint32_t type = read_u32(ptr, swap);
    uint32_t base_type = type & 0xFF;

    switch (base_type) {
        case 1: { // Point
            double x = read_f64(ptr, swap);
            double y = read_f64(ptr, swap);
            update_bounds(minx, miny, maxx, maxy, x, y);
            break;
        }
        case 2: { // LineString
            uint32_t n = read_u32(ptr, swap);
            scan_points(ptr, n, swap, minx, miny, maxx, maxy);
            break;
        }
        case 3: { // Polygon
            uint32_t rings = read_u32(ptr, swap);
            for (uint32_t i = 0; i < rings; ++i) {
                uint32_t n = read_u32(ptr, swap);
                scan_points(ptr, n, swap, minx, miny, maxx, maxy);
            }
            break;
        }
        case 4: case 5: case 6: case 7: { // Multi*, GeometryCollection
            bounds_collection(ptr, end, swap, minx, miny, maxx, maxy);
            break;
        }
        default:
            break;
    }
}

double* fast_batch_bounds(char** wkb_ptrs, int* lens, int count) {
    double* out = new double[count * 4];  // minx, miny, maxx, maxy per row

    #pragma omp parallel for
    for (int i = 0; i < count; ++i) {
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(wkb_ptrs[i]);
        const uint8_t* end = ptr + lens[i];
        double minx = std::numeric_limits<double>::infinity();
        double miny = std::numeric_limits<double>::infinity();
        double maxx = -std::numeric_limits<double>::infinity();
        double maxy = -std::numeric_limits<double>::infinity();

        bounds_wkb_internal(ptr, end, minx, miny, maxx, maxy);

        out[i * 4 + 0] = minx;
        out[i * 4 + 1] = miny;
        out[i * 4 + 2] = maxx;
        out[i * 4 + 3] = maxy;
    }

    return out;
}

void free_result_dbl(double* ptr) {
    delete[] ptr;
}

}

