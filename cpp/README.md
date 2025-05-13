# Spatial Join & Geometry Calibration C++ Modules

These C++ source files implement the low‐level, performance-critical routines for parsing, indexing, and relating geometries in WKB format. Each module exposes a C ABI so that it can be loaded via `ctypes` from Python. OpenMP is used for parallelism, and GEOS C (`geos_c.h`) is leveraged for robust WKT/WKB geometry handling where needed.

---

## Build Prerequisites

- **Compiler**: g++ (≥ C++11), with support for `-fPIC`, `-shared`, and OpenMP (`-fopenmp`)
- **Libraries**: GEOS C development headers and library (`geos_c.h`, `-lgeos_c`)
- **Directory layout**:
```
my_project/
 ├── cpp/           ← all .cpp modules here
 └── lib/           ← compiled .so files will be placed here
```

- **Typical compile command** (from project root):
```bash
g++ -O3 -std=c++11 -fPIC -shared <sources>.cpp -o lib/<output>.so \
    -fopenmp -lgeos_c
```

---

## Module Descriptions

### 1. `candidate_stats.cpp`
**Purpose**: Compute, for each target geometry:
- Sum of source-frequency values (`feature13`)
- Number of source candidates (`feature14`)  
using a CSR (compressed-sparse-row) index.

**API**:
```c
extern "C" void compute_candidate_stats(
    const int* target_ids, int N,
    const int* candidate_offsets, const int* candidate_values,
    const int* frequency,
    float* out_freq_sums,
    int* out_candidate_counts
);
```

**Parallelism**: OpenMP `#pragma omp parallel for` over targets.

---

### 2. `fast_bounds.cpp`
**Purpose**: Compute bounding boxes (`minx, miny, maxx, maxy`) for WKB geometries in batch.

**Helpers**:
- `read_u32()`, `read_f64()`: endian-aware reads
- `update_bounds()`, `scan_points()`: bounding box calculation
- Recursive handling for nested geometries

**API**:
```c
extern "C" double* fast_batch_bounds(char** wkb_ptrs, int* lens, int count);
extern "C" void free_result_dbl(double* ptr);
```

**Output**: Flat array of `double[4 * count]` with bounding boxes.

---

### 3. `fast_count_coords.cpp`
**Purpose**: Count number of coordinates in WKB geometries (Point, LineString, Polygon, Multi*, Collection).

**API**:
```c
extern "C" uint32_t* fast_batch_count_coords(char** wkb_ptrs, int* lens, int count);
extern "C" void free_result_u32(uint32_t* ptr);
```

**Output**: Array of coordinate counts per geometry.

---

### 4. `fast_dimension.cpp`
**Purpose**: Determine topological dimension:
- 0 → Point/MultiPoint
- 1 → Line/MultiLine
- 2 → Polygon/MultiPolygon

**API**:
```c
extern "C" int* fast_batch_dimensions(char** wkb_ptrs, int* lens, int count);
extern "C" void free_result_int(int* ptr);
```

**Output**: Array of integers in {0, 1, 2}

---

### 5. `fast_length.cpp`
**Purpose**: Compute total boundary length of geometries.

**API**:
```c
extern "C" double* fast_batch_length(char** wkb_ptrs, int* lens, int count);
extern "C" void free_result_dbl(double* ptr);
```

**Output**: Array of `double[count]` with line or polygon perimeter lengths.

---

### 6. `grid_bbox_join.cpp`
**Purpose**: Efficient spatial join of bounding boxes using a grid-based spatial index.

**Structure**:
```c
struct IntPair {
    int source_id;
    int target_id;
};
```

**API**:
```c
extern "C" IntPair* grid_bbox_join(
    const float* source_bounds, int n_src,
    const float* target_bounds, int n_tgt,
    float minx, float miny, float maxx, float maxy,
    int grid_x, int grid_y,
    int* out_count
);
extern "C" void free_grid_pairs(IntPair* ptr);
```

**Output**: Array of `(source_id, target_id)` pairs where bounding boxes intersect.

---

### 7. `relate_wkb_u64.cpp`
**Purpose**: Batch relational comparison (DE-9IM) of WKB geometry pairs using GEOS.

**Details**:
- Returns 64-bit encoded DE-9IM patterns
- GEOS handles geometry parsing and relation logic

**API**:
```c
extern "C" uint64_t* relate_batch_wkb_u64(
    char** wkb1_list, int* len1,
    char** wkb2_list, int* len2,
    int count
);
extern "C" void free_result_u64(uint64_t* ptr);
```

**Output**: Array of encoded DE-9IM values as `uint64_t`.

---

### 8. `read_wkt_csv.cpp`
**Purpose**: Detect the geometry column in a delimited WKT CSV, parse geometries, and output WKBs.

**Functionality**:
- Auto-detect geometry column
- Skip invalid/empty/collection geometries
- Parallel WKT parsing

**API**:
```c
extern "C" char** read_wkt_csv_fast(
    const char* filepath,
    char delimiter,
    int* out_count,
    int** out_lengths
);

extern "C" void free_wkt_csv_results(
    char** results, int count, int* lengths
);
```

---

### 9. `wkt_wrapper.cpp`
**Purpose**: Wraps `read_wkt_csv_fast` for direct use via Python's `ctypes`.

**API**:
```c
extern "C" void read_wkb_for_python(
    const char* filepath, char delimiter,
    char*** out_results,
    int* out_count,
    int** out_lengths
);

extern "C" void free_wkb_for_python(
    char** results, int count, int* lengths
);
```

---

## Integration with Python

- Compile each `.cpp` file into a `.so` library under `lib/`
- Use `ctypes.CDLL(lib_path)` in Python wrappers (`fast_geom.py`, `reader.py`, etc.)
- Call C++ routines on NumPy arrays or Python buffers
- Ensures fast computation for millions of spatial operations

---

## Summary

These modules provide high-speed computation for:
- Coordinate counting
- Length and bounding-box calculation
- Spatial filtering via bounding boxes
- Geometry relation evaluation using GEOS
- CSV loading and parsing WKT to WKB

Used together, they enable efficient preprocessing, feature extraction, and spatial join tasks from Python using minimal memory and CPU overhead.
