#include <geos_c.h>
#include <omp.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstring>
#include <iostream>


extern "C" {

/**
 * Tries each column of the first line and detects the one that parses as valid WKT.
 */
int detect_geometry_column(const std::string& line, char delimiter, GEOSContextHandle_t handle) {
    std::istringstream ss(line);
    std::string token;
    int index = 0;
    while (std::getline(ss, token, delimiter)) {
        // Strip quotes if present
        if (!token.empty() && token.front() == '"' && token.back() == '"') {
            token = token.substr(1, token.size() - 2);
        }

        GEOSGeometry* g = GEOSGeomFromWKT_r(handle, token.c_str());
        if (g) {
            GEOSGeom_destroy_r(handle, g);
            return index;
        }
        index++;
    }
    return -1;
}

/**
 * Reads a CSV file (no header) and parses WKT from a detected column into binary WKB.
 */
char** read_wkt_csv_fast(const char* filepath, char delimiter,
                         int* out_count, int** out_lengths)
{
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filepath << std::endl;
        *out_count = 0;
        return nullptr;
    }

    // Detect geometry column using the first data row
    std::string first_line;
    if (!std::getline(file, first_line)) {
        std::cerr << "Empty file or unable to read first row." << std::endl;
        *out_count = 0;
        return nullptr;
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }
    file.close();

    int total = static_cast<int>(lines.size());
    if (total == 0) {
        std::cerr << "No data rows found.\n";
        *out_count = 0;
        return nullptr;
    }

    // Detect geometry column using GEOS
    GEOSContextHandle_t tmp_handle = GEOS_init_r();
    int geom_col = detect_geometry_column(first_line, delimiter, tmp_handle);
    GEOS_finish_r(tmp_handle);

    if (geom_col < 0) {
        std::cerr << "Could not detect geometry column.\n";
        *out_count = 0;
        return nullptr;
    }

    std::cerr << "[INFO] Geometry detected in column: " << geom_col << std::endl;

    // Allocate temporary output buffers
    char** temp_results = new char*[total];
    int* temp_lengths = new int[total];

    for (int i = 0; i < total; i++) {
        temp_results[i] = nullptr;
        temp_lengths[i] = 0;
    }

    // Parallel parsing
    #pragma omp parallel
    {
        GEOSContextHandle_t handle = GEOS_init_r();

        #pragma omp for schedule(static)
        for (int i = 0; i < total; i++) {
            std::istringstream ss(lines[i]);
            std::string token;
            int idx = 0;
            std::string wkt;

            while (std::getline(ss, token, delimiter)) {
                if (idx == geom_col) {
                    wkt = token;
                    break;
                }
                idx++;
            }

            // Strip quotes from WKT if present
            if (!wkt.empty() && wkt.front() == '"' && wkt.back() == '"') {
                wkt = wkt.substr(1, wkt.size() - 2);
            }

            if (wkt.empty()) continue;

            GEOSGeometry* geom = GEOSGeomFromWKT_r(handle, wkt.c_str());
            if (!geom) continue;

            // Skip invalid geometries
            if (!GEOSisValid_r(handle, geom)) {
                GEOSGeom_destroy_r(handle, geom);
                continue;
            }

            // Skip empty geometries
            if (GEOSisEmpty_r(handle, geom)) {
                GEOSGeom_destroy_r(handle, geom);
                continue;
            }

            // Skip geometry collections
            const char* geom_type = GEOSGeomType_r(handle, geom);
            if (geom_type && std::strcmp(geom_type, "GeometryCollection") == 0) {
                GEOSGeom_destroy_r(handle, geom);
                continue;
            }

            size_t size = 0;
            unsigned char* wkb = GEOSGeomToWKB_buf_r(handle, geom, &size);
            GEOSGeom_destroy_r(handle, geom);

            if (!wkb || size == 0) continue;

            char* row_data = new char[size];
            std::memcpy(row_data, wkb, size);

            temp_results[i] = row_data;
            temp_lengths[i] = static_cast<int>(size);

            GEOSFree_r(handle, wkb);
        }

        GEOS_finish_r(handle);
    }

    // Compact results to remove nulls
    std::vector<char*> filtered_results;
    std::vector<int> filtered_lengths;

    for (int i = 0; i < total; i++) {
        if (temp_results[i] != nullptr && temp_lengths[i] > 0) {
            filtered_results.push_back(temp_results[i]);
            filtered_lengths.push_back(temp_lengths[i]);
        }
    }

    // Clean up temporary buffers
    delete[] temp_results;
    delete[] temp_lengths;

    int filtered_count = static_cast<int>(filtered_results.size());
    char** results = new char*[filtered_count];
    int* lengths = new int[filtered_count];

    for (int i = 0; i < filtered_count; i++) {
        results[i] = filtered_results[i];
        lengths[i] = filtered_lengths[i];
    }

    *out_count = filtered_count;
    *out_lengths = lengths;
    return results;
}


/**
 * Cleanup function to free memory from read_wkt_csv_fast
 */
void free_wkt_csv_results(char** results, int count, int* lengths) {
    if (!results) return;
    for (int i = 0; i < count; i++) {
        delete[] results[i];
    }
    delete[] results;
    delete[] lengths;
}

}  // extern "C"

