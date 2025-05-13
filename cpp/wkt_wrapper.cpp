#include <cstring>
#include <vector>

extern "C" {
    char** read_wkt_csv_fast(const char* filepath, char delimiter,
                             int* out_count, int** out_lengths);
    void free_wkt_csv_results(char** results, int count, int* lengths);
}

/**
 * Thin wrapper to use with Python ctypes.
 * This function returns a flat array of pointers to WKB byte strings.
 */
extern "C" void read_wkb_for_python(const char* filepath, char delimiter,
                                    char*** out_results, int* out_count, int** out_lengths)
{
    *out_results = read_wkt_csv_fast(filepath, delimiter, out_count, out_lengths);
}

/**
 * Cleanup wrapper
 */
extern "C" void free_wkb_for_python(char** results, int count, int* lengths) {
    free_wkt_csv_results(results, count, lengths);
}

