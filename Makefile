# ————————————————————————————————
# Makefile for C++ libs + flexible dataset mapping + per-dataset params
# ————————————————————————————————

# Compiler settings
CXX        := g++
CXXFLAGS   := -O3 -fPIC -std=c++11 -fopenmp
LDFLAGS    := -shared
GEOS_LIBS  := -lgeos_c

# Directories
CPP_DIR    := cpp
LIB_DIR    := lib
PYTHON_DIR := python
DATA_DIR   := data

# Dataset selector (default D1; override via: make run DATASET=D2)
DATASET    ?= D1

# ————————————————————————————————
# Map DATASET → source & target filenames
# ————————————————————————————————
SRC_D1 := AREAWATER.csv
TGT_D1 := LINEARWATER.csv

SRC_D2 := AREAWATER.csv
TGT_D2 := ROADS.csv

SRC_D3 := lakes.csv
TGT_D3 := parks.csv

SRC_D4 := parks.csv
TGT_D4 := roads.csv

# ————————————————————————————————
# Map DATASET → BUDGET & QPAIRS
# ————————————————————————————————
BUDGET_D1  := 6310640
QPAIRS_D1  := 2401396

BUDGET_D2  := 15729319
QPAIRS_D2  := 199122

BUDGET_D3  := 19595036
QPAIRS_D3  := 3841922

BUDGET_D4  := 67336808
QPAIRS_D4  := 12145630

# Pick the right file & params for the current DATASET
SRC_FILE  := $(DATA_DIR)/$(SRC_$(DATASET))
TGT_FILE  := $(DATA_DIR)/$(TGT_$(DATASET))
BUDGET    := $(BUDGET_$(DATASET))
QPAIRS    := $(QPAIRS_$(DATASET))

# Global algorithm defaults (can still override on command line)
RECALL    ?= 0.90
SAMPLER   ?= hashing
THRESHOLD ?= ensemble_multi

# ————————————————————————————————
# C++ sources → shared libraries
# ————————————————————————————————
WKT_SRCS   := \
    $(CPP_DIR)/read_wkt_csv.cpp \
    $(CPP_DIR)/wkt_wrapper.cpp

OTHER_SRCS := \
    grid_bbox_join.cpp       \
    relate_wkb_u64.cpp       \
    fast_count_coords.cpp    \
    fast_length.cpp          \
    fast_bounds.cpp          \
    fast_dimension.cpp       \
    candidate_stats.cpp

SOFILES    := $(LIB_DIR)/libwkt.so \
              $(patsubst %.cpp,$(LIB_DIR)/lib%.so,$(OTHER_SRCS))

.PHONY: all install run run-all clean

all: $(LIB_DIR) $(SOFILES)

# Ensure lib/ exists
$(LIB_DIR):
	mkdir -p $(LIB_DIR)

#
# Build libwkt.so from both read_wkt_csv.cpp + wkt_wrapper.cpp
#
$(LIB_DIR)/libwkt.so: $(WKT_SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(GEOS_LIBS)


# Build librelate_wkb_u64.so—and link in GEOS_C for GEOSWKBReader_*
$(LIB_DIR)/librelate_wkb_u64.so: $(CPP_DIR)/relate_wkb_u64.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(GEOS_LIBS)

#
# Build the rest via a pattern rule
#
$(LIB_DIR)/lib%.so: $(CPP_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# ————————————————————————————————
# Python deps
# ————————————————————————————————
install:
	pip install -r requirements.txt

# ————————————————————————————————
# Run on a single dataset
# ————————————————————————————————
run: all install
	@echo "→ Running calibration on dataset $(DATASET)"
	@echo "    source: $(SRC_FILE)"
	@echo "    target: $(TGT_FILE)"
	@echo "    budget: $(BUDGET), qPairs: $(QPAIRS)"
	PYTHONPATH=$(PYTHON_DIR):$(LIB_DIR) \
	  python3 $(PYTHON_DIR)/calibration.py \
	    --src             $(SRC_FILE) \
	    --tgt             $(TGT_FILE) \
	    --budget          $(BUDGET) \
	    --qPairs          $(QPAIRS) \
	    --recall          $(RECALL) \
	    --sampling_method $(SAMPLER) \
	    --threshold_method $(THRESHOLD)

# ————————————————————————————————
# Loop through D1…D4 in sequence
# ————————————————————————————————
run-all:
	@for ds in D1 D2 D3 D4 ; do \
	  echo; echo "=== Dataset $$ds ==="; \
	  $(MAKE) run DATASET=$$ds; \
	done; echo

# ————————————————————————————————
# Clean up
# ————————————————————————————————
clean:
	rm -rf $(LIB_DIR)/*.so
