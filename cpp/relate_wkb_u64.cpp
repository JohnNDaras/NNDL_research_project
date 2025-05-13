#include <geos_c.h>
#include <omp.h>
#include <cstring>
#include <cstdint>
#include <cstdlib>

extern "C" {

// Encode DE-9IM string into uint64_t (first 8 chars, packed)
inline uint64_t encode_de9im(const char* s) {
    uint64_t val = 0;
    for (int i = 0; i < 8 && s[i]; ++i) {
        val |= ((uint64_t)(uint8_t)s[i]) << (i * 8);
    }
    return val;
}

// Batch relate using binary WKB, return encoded DE-9IM as uint64_t
uint64_t* relate_batch_wkb_u64(char** wkb1_list, int* len1, char** wkb2_list, int* len2, int count) {
    uint64_t* results = new uint64_t[count];

    #pragma omp parallel
    {
        GEOSContextHandle_t handle = GEOS_init_r();
        GEOSWKBReader* reader = GEOSWKBReader_create_r(handle);

        #pragma omp for schedule(static)
        for (int i = 0; i < count; ++i) {
            const unsigned char* data1 = (const unsigned char*)wkb1_list[i];
            const unsigned char* data2 = (const unsigned char*)wkb2_list[i];

            GEOSGeometry* g1 = GEOSWKBReader_read_r(handle, reader, data1, len1[i]);
            GEOSGeometry* g2 = GEOSWKBReader_read_r(handle, reader, data2, len2[i]);

            if (!g1 || !g2) {
                results[i] = 0;  // Use 0 to represent error
            } else {
                char* rel = GEOSRelate_r(handle, g1, g2);
                results[i] = encode_de9im(rel);
                GEOSFree_r(handle, rel);
            }

            if (g1) GEOSGeom_destroy_r(handle, g1);
            if (g2) GEOSGeom_destroy_r(handle, g2);
        }

        GEOSWKBReader_destroy_r(handle, reader);
        GEOS_finish_r(handle);
    }

    return results;
}

void free_result_u64(uint64_t* result) {
    delete[] result;
}
}
