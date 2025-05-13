# NNDL_research_project

# Spatial Join Recall Estimation – Experimental Evaluation

This project evaluates various recall estimation methods under constrained budgets for spatial join problems. The experiments cover four different real-world datasets (D1–D4), several recall targets (0.70, 0.80, 0.90), and five different methods:

- **Proposed**: The main method introduced by this project.
- **Ensemble-1**: An ensemble approach using multiple estimators.
- **QuantCI**: A quantile-based confidence interval.
- **Wilson-hash**: Wilson interval with hashing-based sampling.
- **Wilson-rnd**: Wilson interval with random sampling.

---

## How to Run Experiments

Make sure you have built all C++ shared libraries and set up the Python environment.

### Step 1: Build shared libraries
```bash
make all
make install
```

### Step 2: Run experiments using `make run`
Use the dataset shortcut (`D1`, `D2`, etc.) to launch a specific configuration. For example:
```bash
make run DATASET=D3
```

To evaluate all datasets sequentially:
```bash
make run-all
```

Internally, `make run` will point to the correct files and arguments for each dataset. Here's how dataset-to-file mapping is handled:

| Dataset | Source File      | Target File      | Budget   | QPairs   |
|---------|------------------|------------------|----------|----------|
| D1      | `AREAWATER.csv`  | `LINEARWATER.csv`| 6310640  | 2401396  |
| D2      | `AREAWATER.csv`  | `ROADS.csv`      | 15729319 | 199122   |
| D3      | `lakes.csv`      | `parks.csv`      | 19595036 | 3841922  |
| D4      | `parks.csv`      | `roads.csv`      | 67336808 |12145630  |

---

## Experimental Results

We evaluate the variance and accuracy of recall estimates under different recall targets across all four datasets.

### Standard Deviation of Recall Across Methods
This figure shows the average standard deviation of recall over multiple runs for each method and target level.

![Std Dev Combined](std_combined.png)

---

###  Mean Recall for Each Method Across Datasets
This figure shows the mean achieved recall for each method across all datasets at different target levels.

![Mean Recall Combined](mean_combined.png)

---

## 📋 Detailed Tables

### Table 1: Achieved Recall (μ ± σ) — Dataset D1

| Target | QuantCI         | IVW-1          | Wilson-rnd      | Proposed        |
|--------|------------------|----------------|------------------|------------------|
| 0.70   | 0.693 ± 0.007    | 0.696 ± 0.001  | 0.701 ± 0.009    | 0.698 ± 0.001    |
| 0.80   | 0.799 ± 0.003    | 0.798 ± 0.002  | 0.798 ± 0.004    | 0.799 ± 0.001    |
| 0.90   | 0.901 ± 0.002    | 0.900 ± 0.002  | 0.902 ± 0.002    | 0.900 ± 0.002    |

---

### Table 2: Achieved Recall (μ ± σ) — Dataset D2

| Target | QuantCI         | IVW-1          | Wilson-rnd      | Proposed        |
|--------|------------------|----------------|------------------|------------------|
| 0.70   | 0.701 ± 0.012    | 0.694 ± 0.007  | 0.701 ± 0.062    | 0.699 ± 0.002    |
| 0.80   | 0.793 ± 0.003    | 0.786 ± 0.010  | 0.777 ± 0.036    | 0.798 ± 0.001    |
| 0.90   | 0.889 ± 0.015    | 0.892 ± 0.002  | 0.891 ± 0.035    | 0.892 ± 0.002    |

---

### Table 3: Achieved Recall (μ ± σ) — Dataset D3

| Target | QuantCI         | IVW-1          | Wilson-rnd      | Proposed        |
|--------|------------------|----------------|------------------|------------------|
| 0.70   | 0.703 ± 0.014    | 0.694 ± 0.011  | 0.713 ± 0.122    | 0.701 ± 0.005    |
| 0.80   | 0.800 ± 0.006    | 0.787 ± 0.012  | 0.785 ± 0.025    | 0.799 ± 0.001    |
| 0.90   | 0.891 ± 0.024    | 0.893 ± 0.002  | 0.906 ± 0.050    | 0.901 ± 0.013    |

---

### Table 4: Achieved Recall (μ ± σ) — Dataset D4

| Target | QuantCI         | IVW-1          | Wilson-rnd      | Proposed        |
|--------|------------------|----------------|------------------|------------------|
| 0.70   | 0.691 ± 0.007    | 0.694 ± 0.002  | 0.691 ± 0.005    | 0.694 ± 0.001    |
| 0.80   | 0.794 ± 0.002    | 0.790 ± 0.001  | 0.791 ± 0.002    | 0.793 ± 0.001    |
| 0.90   | 0.891 ± 0.003    | 0.890 ± 0.001  | 0.893 ± 0.002    | 0.891 ± 0.002    |

---

## Notes

- All methods were tested using the same sampling budget per dataset.
- Variance was calculated using standard deviation over 3–5 independent runs.
- "Proposed" consistently exhibits lower variance and more stable recall.

For questions, refer to the full implementation and code documentation in the respective Python and C++ components.



