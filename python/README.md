# Spatial Join & Geometry Calibration Toolkit

This repository provides a high-performance pipeline for large-scale spatial joins and geometry-based relationship calibration. The core of the system is a C++ library (exposed via `ctypes`) for extremely fast WKB parsing, bounding-box indexing, and DE-9IM relating, combined with a Python framework for sampling, model training, and threshold calibration.

---

## Table of Contents

1. [Quickstart](#quickstart)  
2. [Project Layout](#project-layout)  
3. [Module Overview](#module-overview)  
   - [`reader.py`](#readerpy)  
   - [`candidate_stats.py`](#candidate-statspy)  
   - [`fast_geom.py`](#fast-geumpy)  
   - [`de9im_patterns.py`](#de9im-patternspy)  
   - [`related_geoms.py`](#related-geometrypy)  
4. [Workflow Example](#workflow-example)  
5. [Development & Extension](#development--extension)  

---

## Quickstart

1. **Build & install**  
   ```bash
   make all         # compiles C++ libraries into lib/
   make install     # installs Python dependencies
   ```
2. **Run** on dataset “D1” (defaults):  
   ```bash
   make run
   ```
3. **Run** on all datasets in sequence:  
   ```bash
   make run-all
   ```

---

## Project Layout

```
my_project/
├── Makefile
├── requirements.txt
├── README.md
├── data/                   # your input CSVs/WKT files
├── lib/                    # compiled .so libraries
├── cpp/                    # C++ source files
└── python/                 # Python modules
    ├── reader.py
    ├── candidate_stats.py
    ├── fast_geom.py
    ├── de9im_patterns.py
    ├── related_geoms.py
    └── calibration.py      # main driver (calls the above)
```

---

## Module Overview

### `reader.py`

**Purpose:**  
Wraps the C++ WKT‐to‐WKB converter via `ctypes`. Reads a CSV file, auto-detects the geometry column, and returns a Python list of raw WKB byte strings.

**Key API:**

```python
class CsvReader:
    def __init__(self, lib_path: str = None):
        # loads lib/libwkt.so by default

    def readAllEntities(self, filepath: str, delimiter: str = '\t') -> List[bytes]:
        """
        Reads up to N rows of the CSV at `filepath`,
        auto-detects which column holds valid WKT,
        converts each WKT → WKB,
        and returns a list of bytes objects.
        """
```

**Example:**

```python
from reader import CsvReader

reader = CsvReader()
wkb_list = reader.readAllEntities("data/AREAWATER.csv", delimiter=",")
```

---

### `candidate_stats.py`

**Purpose:**  
Computes two features for each target geometry ID:  
1. Sum of source-frequency values over its candidate list.  
2. Number of candidate sources.  

It uses a CSR representation of the candidate index (built in Python) and offloads the inner loop to C++/OpenMP.

**Key API:**

```python
class CandidateStats:
    def __init__(self, lib_path: str = None):
        # loads lib/libcandidate_stats.so by default

    def compute(self,
                target_ids: List[int],
                candidate_offsets: np.ndarray,
                candidate_values: np.ndarray,
                frequency: List[int]
               ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
           (freq_sums, candidate_counts)
         each as a NumPy array of length len(target_ids).
        """
```

**Example:**

```python
from candidate_stats import CandidateStats

cs = CandidateStats()
freq_sums, counts = cs.compute(
    target_ids=np.array([0,1,2]),
    candidate_offsets=np.array([0,3,6,9]),
    candidate_values=np.array([…]),
    frequency=[5,2,7,…]
)
```

---

### `fast_geom.py`

**Purpose:**  
Provides extremely fast batch operations on WKB geometries via six C++ libraries:

- **Count coordinates** (`libfast_count_coords.so`)  
- **Compute length**     (`libfast_length.so`)  
- **Compute bounds**     (`libfast_bounds.so`)  
- **Compute dimension**  (`libfast_dimension.so`)  
- **DE-9IM relate**      (`librelate_wkb_u64.so`)  
- **Grid-based bbox join** (`libgrid_bbox_join.so`)

All methods accept Python lists of WKB bytes and return NumPy arrays.

**Key API:**

```python
class FastGeom:
    def __init__(…):
        # loads all six .so files by default

    def get_num_of_points(self, wkb_list: List[bytes]) -> np.ndarray
    def get_lengths(self, wkb_list: List[bytes]) -> np.ndarray
    def get_bounds(self, wkb_list: List[bytes]) -> np.ndarray  # shape (N,4)
    def get_dimensions(self, wkb_list: List[bytes]) -> np.ndarray
    def relate(self, wkb1: List[bytes], wkb2: List[bytes]) -> np.ndarray
        # returns uint64 bit‐packed DE-9IM codes, length N
    def grid_bbox_intersect(self,
                            source_bounds: np.ndarray,
                            target_bounds: np.ndarray,
                            extent: Tuple[float],
                            grid_x: int, grid_y: int
                           ) -> np.ndarray
        # returns array of shape (M,2) with (source_id, target_id) candidate pairs
```

**Example:**

```python
from fast_geom import FastGeom

fg = FastGeom()
lengths = fg.get_lengths(wkb_list)
bounds  = fg.get_bounds(wkb_list)
pairs   = fg.grid_bbox_intersect(bounds, other_bounds, extent, 64, 64)
```

---

### `de9im_patterns.py`

**Purpose:**  
Implements fully vectorized DE-9IM pattern matching in pure NumPy. Converts 9-character DE-9IM strings into 9-column bitmasks, then provides:

- **`Pattern`**: require each position to match a set of allowed bits  
- **`AntiPattern`**: invert a pattern match  
- **`NOrPattern`**: match any of several patterns  

and supplies common spatial predicates:

```python
contains, crosses_lines, crosses_1, crosses_2,
disjoint, equal, intersects_de9im,
overlaps1, overlaps2,
touches, within, covered_by, covers
```

**Key API:**

```python
def encode_de9im_strings(de9im_array: List[str]) -> np.ndarray  # (N,9)

class Pattern(pattern_str: str):
    def matches_array(self, encoded: np.ndarray) -> np.ndarray[bool]

class AntiPattern(…)
class NOrPattern(…)
```

**Example:**

```python
from de9im_patterns import encode_de9im_strings, contains

encoded = encode_de9im_strings(['FF*FF****', 'T*F**F***'])
mask = contains.matches_array(encoded)
```

---

### `related_geoms.py`

**Purpose:**  
Leverages `FastGeom.relate(...)` and `de9im_patterns` to classify every candidate pair into standard spatial relations.  Tracks per‐relation counts and can print precision/recall statistics.

**Core class:**

```python
class RelatedGeometries:
    def __init__(self, qualifyingPairs: int):
        # qualifyingPairs = total number of true pairs expected

    def verifyRelations(self,
                        geomIds1: List[int],
                        geomIds2: List[int],
                        sourceGeoms: List[bytes],
                        targetGeoms: List[bytes]
                       ) -> np.ndarray[bool]:
        """
        1. Calls FastGeom.relate(...) to get uint64 DE-9IM codes for each pair.
        2. Decodes each code into a 9-char string.
        3. Vectorizes pattern matching to detect:
           intersects, within, covered_by, overlaps, crosses, equal,
           touches, contains, covers.
        4. Updates internal counters & per-relation ID lists.
        Returns a boolean mask of length N indicating which pairs are “related.”
        """
```

Other helpers:

- `.addContains`, `.addWithin`, etc. — record IDs  
- `.getNoOfContains()`, etc. — inspect counts  
- `.print()` — display summary stats  

**Example:**

```python
from related_geoms import RelatedGeometries

rg = RelatedGeometries(qualifyingPairs=10000)
mask = rg.verifyRelations(ids1, ids2, src_wkbs, tgt_wkbs)

print("Found", rg.getNoOfContains(), "contains")
rg.print()
```

---

## Workflow Example

1. **Load raw WKB** via `CsvReader.readAllEntities`.  
2. **Compute bounds**: `FastGeom.get_bounds`.  
3. **Grid‐index join**: `FastGeom.grid_bbox_intersect` → candidate pairs.  
4. **Calibration sampling** (Python logic).  
5. **Initial relationship verification**:  
   ```python
   rg = RelatedGeometries(qPairs)
   mask = rg.verifyRelations(src_ids, tgt_ids, src_wkb, tgt_wkb)
   ```
6. **Feature extraction**:  
   - areas, intersection area, block counts  
   - `CandidateStats.compute` for candidate counts  
   - `FastGeom.get_num_of_points/lengths`  
7. **Train a PyTorch model** (in `calibration.py`).  
8. **Threshold calibration** (Wilson / Clopper–Pearson / QuantCI / ensemble).  
9. **Final filtering & verification** (calls `verifyRelations` again).

---

## Development & Extension

- **Adding a new DE-9IM relation**:  
  1. Add its pattern to `de9im_patterns.py` (via `Pattern` or `NOrPattern`).  
  2. In `related_geoms.py`, compute a `mask_xyz = new_pattern.matches_array(encoded)` and call `.addXyz` in the mask loop.

- **Tuning grid size**: modify `thetaX/Y` logic in `calibration.py::estimate_grid_size`.

- **Parallelism**:  
  - C++ uses OpenMP; tune `-fopenmp` or schedule.  
  - Python sampling/verification can be multi-process if needed.

- **Error handling**: Currently invalid geometries produce warnings. You can capture these in `self.exceptions`.

---

With this setup, you get **C++-level speed** for low-level geometry ops, combined with **Python flexibility** for sampling, training, and calibration. Enjoy!
