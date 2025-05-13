
---

## Module Overview




---




# `thresholds.py`

## Overview

`thresholds.py` implements several **threshold‐selection algorithms** for achieving a user‐specified recall target R* on a set of positively‐labeled scores. Each function takes as input an array of scores for verified positives (and optionally labels), and returns the smallest score threshold whose corresponding lower‐bound recall (or estimated recall) meets or exceeds R*.

---

## Table of Contents

1. [Dependencies](#dependencies)  
2. [Functions](#functions)  
   - [`threshold_recall_wilson`](#1-threshold_recall_wilson)  
   - [`threshold_recall_confidence`](#2-threshold_recall_confidence-clopper--pearson)  
   - [`threshold_quant_ci`](#3-threshold_quant_ci)  
3. [Usage Examples](#usage-examples)  
4. [Parameter Reference](#parameter-reference)  

---

## Dependencies

- **NumPy**  
- **SciPy** (for `scipy.stats.norm` and `scipy.stats.beta`)

Install via:

```bash
pip install numpy scipy
```

---

## Functions

### 1. `threshold_recall_wilson`

```python
def threshold_recall_wilson(
        probs_pos: Sequence[float],
        target_recall: float = 0.90,
        alpha: float = 0.05,
        verbose: bool = True
    ) -> float:
```

**Purpose:**  
Finds the smallest threshold \( \tau \) such that the one‐sided Wilson score lower bound on recall


![Image](https://github.com/user-attachments/assets/f2dc415e-82ed-4669-9514-42e1ad28f326)


where

![Image](https://github.com/user-attachments/assets/7f9e2ea9-f4ea-4945-bc0d-2828b985d6cc)

meets >= R*

**Inputs:**  
- `probs_pos`: array of scores for true positives  
- `target_recall`: desired recall \(R^\star\)  
- `alpha`: significance level (default 0.05 → 95% lower bound)  
- `verbose`: print threshold & diagnostics  

**Returns:**  
- `threshold` (float)

---

### 2. `threshold_recall_confidence` (Clopper–Pearson)

```python
def threshold_recall_confidence(
        probs_pos: Sequence[float],
        target_recall: float = 0.90,
        alpha: float = 0.05,
        verbose: bool = True
    ) -> float:
```

**Purpose:**  
Uses the Clopper–Pearson exact binomial lower‐confidence bound

![Image](https://github.com/user-attachments/assets/6951626a-c527-4b8b-813b-f097657a916b)

to select the smallest threshold achieving L_k >= R*.

**Inputs & Returns:** same as Wilson method.

---

### 3. `threshold_quant_ci`

```python
def threshold_quant_ci(
        scores_all: Sequence[float],
        labels: Sequence[int],
        target_recall: float = 0.90,
        alpha: float = 0.05,
        verbose: bool = True
    ) -> float:
```

**Purpose:**  
Implements a **Horvitz–Thompson–style** sequential CI:  
- Sorts all sample scores in descending order.  
- Iterates through ranks \(k=1\ldots\) until the normal‐approximate lower bound:

![Image](https://github.com/user-attachments/assets/d2917296-6dd5-4653-8d26-69f7069d2fa5)

meets R*, where 

![Image](https://github.com/user-attachments/assets/36acb145-ba36-4db6-824f-7d3bc1704c03)


**Inputs:**  
- `scores_all`: array of predicted probabilities  
- `labels`: binary labels for each pair  

**Returns:**  
- `thr`: minimum probability threshold satisfying the lower-bound constraint

---

## Usage Examples

```python
import numpy as np
from thresholds import threshold_recall_wilson, threshold_recall_confidence, threshold_quant_ci

# Example positive scores
pos_scores = np.random.beta(2,5, size=1000)

# 1. Wilson lower bound
tau_wilson = threshold_recall_wilson(
    probs_pos=pos_scores,
    target_recall=0.90,
    alpha=0.05
)

# 2. Clopper–Pearson
tau_cp = threshold_recall_confidence(
    probs_pos=pos_scores,
    target_recall=0.80,
    alpha=0.10
)

# 3. QuantCI (requires labels)
all_scores = np.concatenate([pos_scores, np.random.rand(2000)])
all_labels = np.array([1]*len(pos_scores) + [0]*2000)
tau_quant = threshold_quant_ci(
    scores_all=all_scores,
    labels=all_labels,
    target_recall=0.85,
    alpha=0.05
)
```

---

## Parameter Reference

| Parameter        | Description                                                         |
|------------------|---------------------------------------------------------------------|
| `probs_pos`      | 1D array of scores for known positives                              |
| `scores_all`     | 1D array of scores for entire calibration sample                   |
| `labels`         | 1D binary array of ground‐truth labels for calibration sample      |
| `target_recall`  | Desired recall \(R^\star\), e.g. 0.70, 0.80, 0.90                   |
| `alpha`          | Significance level for lower bound (e.g. 0.05 → 95% confidence)     |
| `verbose`        | If `True`, prints threshold and diagnostic messages                 |

---

# README for `ensemble.py`

## Overview

`ensemble.py` provides two **ensemble calibration** routines:

1. `ensemble_threshold`:  
   Inverse-variance weighted fusion of 4 base methods (CP, Jeffreys, Wilson, exact quantile) on a single subsample.

2. `ensemble_threshold_multi`:  
   Applies `ensemble_threshold` on multiple subsamples and fuses their results using `min`, `mean`, `median`, or `inverse-variance`.

---

## Dependencies

```bash
pip install numpy scipy
```

---

## `ensemble_threshold`

```python
def ensemble_threshold(
        pos_scores: np.ndarray,
        target_recall: float = 0.90,
        alpha: float = 0.05,
        n_boot: int = 200,
        random_state: int = 42,
        verbose: bool = True
    ) -> float
```

**Returns:**  
Inverse-variance weighted score threshold from Clopper–Pearson, Jeffreys, Wilson, and exact quantile estimators.

Each rule is run over `n_boot` bootstrap resamplings of the positive scores. The ensemble combines them with:

![Image](https://github.com/user-attachments/assets/24f1dd77-b3ba-43d3-9dce-3be6c08db21c)

---

## `ensemble_threshold_multi`

```python
def ensemble_threshold_multi(
        pos_scores: np.ndarray,
        target_recall: float = 0.90,
        alpha: float = 0.05,
        n_boot: int = 200,
        K: int = 9,
        subsample_fr: float = 0.80,
        random_state: int = 42,
        fuse_method: str = "min",
        verbose: bool = True
    ) -> Tuple[float, List[float]]:
```

**Returns:**  
- Final fused threshold (float)  
- List of thresholds across `K` subsamples  

Fusing options:
- `"min"` (default): most conservative  
- `"median"`  
- `"mean"`  
- `"inverse-variance"`

---

## Usage

```python
from ensemble import ensemble_threshold, ensemble_threshold_multi

tau_single = ensemble_threshold(scores, target_recall=0.90)

tau_final, all_subs = ensemble_threshold_multi(
    scores,
    target_recall=0.90,
    K=9,
    fuse_method="min"
)
```

---

## Parameters Summary

| Param             | Meaning                                                      |
|------------------|--------------------------------------------------------------|
| `pos_scores`      | Array of verified positive scores                           |
| `target_recall`   | Desired recall (e.g. 0.90)                                   |
| `alpha`           | Confidence level (e.g. 0.05 for 95%)                         |
| `n_boot`          | Number of bootstrap samples per method                      |
| `K`               | Number of independent subsamples in multi-ensemble          |
| `subsample_fr`    | Fraction of positives to sample per subsample               |
| `fuse_method`     | How to combine K thresholds: `"min"`, `"mean"`, `"median"`  |
| `verbose`         | Print detailed output                                       |

---

# `calibration.py`

## Overview

`calibration.py` implements a complete pipeline for **exact‐recall threshold calibration** in large‐scale spatial entity matching. It ingests two polygon datasets (source and target), filters candidate pairs via an equigrid index, trains a small neural “ranker” on a bootstrap sample, builds a reproducible calibration set (either random or xxHash‐stratified), applies a variety of classical and ensemble threshold rules, and finally verifies until a user‐specified budget or recall target is met.


## `calibration_Based_Algorithm`

Encapsulates data loading, preprocessing, model training, sampling, threshold estimation, and verification.

---

## Table of Contents

- [Features](#features)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Class: calibration_Based_Algorithm](#class-calibration_based_algorithm)
- [Threshold Methods](#threshold-methods)
- [Configuration & Hyper-parameters](#configuration--hyper-parameters)
- [Performance & Logging](#performance--logging)
- [Extending & Troubleshooting](#extending--troubleshooting)

---

## Features

- **Spatial Filtering via equigrid MBR index → CSR arrays**
- **Deterministic Sampling**
  - Random subsample
  - xxHash‐decile‐stratified calibration set
- **Neural Ranker**
  - 16-dim input → 128 → 64 → 1 sigmoid
  - Trained on 1000 labeled pairs (500 positive / 500 negative)
- **Threshold Calibration**
  - Clopper–Pearson, Wilson, Jeffreys, exact quantile
  - Inverse‐variance ensemble
  - Min‐fusion across 9 stratified subsamples
- **Budget-constrained Verification**
  - DE-9IM relations via `RelatedGeometries`
  - Reports true positives until budget exhausted
- **Fully TPU-friendly** via `torch_xla`

---

## Dependencies

- Python 3.8+
- NumPy
- Pandas
- SciPy
- scikit-learn
- Shapely
- xxhash
- PyTorch + torch_xla
- `CsvReader`, `RelatedGeometries`, `FastGeom`, `CandidateStats` (project-specific utility modules)

### Install via:

```bash
pip install numpy pandas scipy scikit-learn shapely xxhash torch torchvision torch_xla
```

---

## Usage

```python
from calibration import calibration_Based_Algorithm

# Define parameters
algo = calibration_Based_Algorithm(
    budget           = 140_000,
    qPairs           = 100_000,
    delimiter        = '\t',
    sourceFilePath   = 'data/source.tsv',
    targetFilePath   = 'data/target.tsv',
    target_recall    = 0.90,
    sampling_method  = 'hashing',     # or 'random'
    threshold_method = 'ensemble_multi'  # or 'QuantCI', 'Clopper_Pearson', 'wilson', 'ensemble'
)
algo.applyProcessing()
```

---

## File Structure

```
recall-exact-calib/
├── calibration.py         # Main pipeline class
├── csv_reader.py          # CsvReader: loads geometries as WKB/polygons
├── related_geometries.py  # RelatedGeometries: efficient DE-9IM verifier
├── fast_geom.py           # FastGeom: bounds, lengths, point counts
├── candidate_stats.py     # CandidateStats: per-target co-occurrence stats
├── requirements.txt       # Pinned dependencies
└── notebooks/             # Example Colab notebooks
```

---

## Class: `calibration_Based_Algorithm`

### `__init__(…)`

```python
def __init__(self, budget, qPairs, delimiter, sourceFilePath, targetFilePath, target_recall, sampling_method, threshold_method)
```

- `budget`: max pairs to verify
- `qPairs`: DE-9IM comparisons in bootstrap
- `delimiter`: e.g. `'\t'` for TSV
- `sourceFilePath` / `targetFilePath`: input geometry files
- `target_recall`: desired recall (e.g. 0.90)
- `sampling_method`: `'hashing'` or `'random'`
- `threshold_method`: `'QuantCI'`, `'Clopper_Pearson'`, `'wilson'`, `'ensemble'`, `'ensemble_multi'`

### `applyProcessing()`

Full pipeline controller:

1. `setThetas()` — compute equigrid cell size  
2. `preprocessing()` — indexing, filtering, min/max stats  
3. `trainModel()` — 500+500 training  
4. `build_calibration_sample()` (if hashing)  
5. `verification()` — scoring + threshold + review

Prints:

```yaml
Indexing Time     : hh:mm:ss
Initialization    : hh:mm:ss
Training Time     : hh:mm:ss
Sampling Time     : hh:mm:ss  # if hashing
Verification Time : hh:mm:ss
```

---

## Preprocessing

### `setThetas()`

- Reads source bounds using `FastGeom`
- Averages MBR width/height → `thetaX`, `thetaY`

### `estimate_grid_size(n)`

Returns square grid size `dim × dim` (between 16–256)

### `build_candidate_csr(indices, num_targets)`

CSR representation:
- `offsets`: length = `num_targets + 1`
- `values`: sourceIds per targetId

### `preprocessing()`

- Uses hashing or random for `sample_ids`, `calibration_sample_ids`
- Computes bounding boxes, areas, blocks, points, lengths
- Applies grid snapping → intersecting tile IDs
- Filters valid source-target MBR pairs
- Populates calibration and training sets

---

## Sampling

### `build_calibration_sample(max_pairs=250_000, seed=2026)`

- Scores all filtered pairs with `predict_in_batches()`
- Assigns score decile
- Uses `xxhash(id ∥ decile)` for deterministic selection
- Picks top `k` pairs across all deciles

---

## Model Training

### `create_model(input_dim)`

- `Linear(16→128)` → ReLU → Dropout(0.3) → BatchNorm  
- `Linear(128→64)` → ReLU → Dropout(0.5) → BatchNorm  
- `Linear(64→1)` → Sigmoid

### `train_model(X, y)`

- Validates data
- Splits 90/10 for validation
- Trains using BCE + Adam (lr=1e-3), early stopping

### `trainModel()`

- Shuffles and verifies pairs
- Uses `RelatedGeometries` to select 500 pos/neg
- Featurizes and trains the model

---

## Prediction

### `predict(X)`

Single-batch inference on TPU

### `predict_in_batches(X, batch_size=8192)`

Batched TPU inference for large datasets

---

## Feature Extraction

### `get_feature_vector(sourceIds, targetIds)`

Constructs 16D vector per (sId, tId) pair:

- MBR areas
- Intersection area
- Co-occurrence stats
- Point counts
- Lengths
- # Neighbors

→ Scaled to [0, 10000] for each feature

### `getNoOfBlocks1(envelopes)`

Computes # grid blocks each geometry spans

---

## Verification

### `verification()`

- Shuffles calibration pairs
- Verifies positives with `RelatedGeometries`
- Gets prediction probabilities
- Applies selected threshold method:
  - `threshold_quant_ci`
  - `threshold_recall_confidence`
  - `threshold_recall_wilson`
  - `ensemble_threshold`
  - `ensemble_threshold_multi`
- Filters top candidates exceeding threshold
- Verifies final top-k pairs until budget is met

---

## Threshold Methods

- `QuantCI`: unbiased recall estimator  
- `Clopper_Pearson`: exact binomial lower bound  
- `Wilson`: normal approx. lower bound  
- `ensemble`: inverse-variance fusion of 4 rules  
- `ensemble_multi`: `min()` across 9 stratified ensemble runs

Each returns a scalar threshold on probability.

---

## Configuration & Hyper-parameters

| Parameter                 | Value         | Description                         |
|--------------------------|---------------|-------------------------------------|
| `CLASS_SIZE`             | 500           | Number of positive/negative labels  |
| `SAMPLE_SIZE`            | 50,000        | Bootstrap sample size               |
| Calibration sample size  | 250,000       | Pairs for threshold estimation      |
| Subsamples (multi)       | 9             | In ensemble multi                   |
| Bootstraps per rule      | 200           | For variance estimation             |
| Batch size               | 8,192         | Inference batch size                |
| Learning rate            | 1e-3          | Adam optimizer                      |
| Epochs / Early-stop      | 30 / 3        | Max and patience                    |


---

## Extending & Troubleshooting

- Replace `xla_device()` with `torch.device("cpu"/"cuda")` to run on CPU/GPU.
- Ensure utility modules return expected data types (NumPy, Shapely).
- Add new threshold methods by extending `verification()`.
- Reduce calibration size or batch size to save memory.
- Insert `print()` or use `logging` to trace/debug stages.

---

# Spatial Join & Geometry Calibration Toolkit

This repository provides a high-performance pipeline for large-scale spatial joins and geometry-based relationship calibration. The core of the system is a C++ library (exposed via `ctypes`) for extremely fast WKB parsing, bounding-box indexing, and DE-9IM relating, combined with a Python framework for sampling, model training, and threshold calibration.



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
