// grid_bbox_join.cpp
#include <vector>
#include <unordered_map>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <cmath>

extern "C" {

struct IntPair {
    int source_id;
    int target_id;
};

inline bool bboxes_intersect(const float* a, const float* b) {
    return !(a[2] < b[0] || a[0] > b[2] || a[3] < b[1] || a[1] > b[3]);
}

// Spatial hash key (1D)
inline int grid_hash(int gx, int gy, int grid_x) {
    return gy * grid_x + gx;
}

IntPair* grid_bbox_join(
    const float* source_bounds, int n_src,
    const float* target_bounds, int n_tgt,
    float minx, float miny, float maxx, float maxy,
    int grid_x, int grid_y,
    int* out_count
) {
    float cell_w = (maxx - minx) / grid_x;
    float cell_h = (maxy - miny) / grid_y;

    // === 1. Build spatial hash grid of source boxes ===
    std::unordered_map<int, std::vector<int>> grid;

    for (int i = 0; i < n_src; ++i) {
        const float* b = source_bounds + i * 4;
        int gx_min = std::max(0, (int)((b[0] - minx) / cell_w));
        int gx_max = std::min(grid_x - 1, (int)((b[2] - minx) / cell_w));
        int gy_min = std::max(0, (int)((b[1] - miny) / cell_h));
        int gy_max = std::min(grid_y - 1, (int)((b[3] - miny) / cell_h));

        for (int gx = gx_min; gx <= gx_max; ++gx) {
            for (int gy = gy_min; gy <= gy_max; ++gy) {
                int key = grid_hash(gx, gy, grid_x);
                grid[key].push_back(i);
            }
        }
    }

    // === 2. Query phase (Parallel) ===
    std::vector<IntPair> all_pairs;

    #pragma omp parallel
    {
        std::vector<IntPair> local_pairs;

        #pragma omp for schedule(static)
        for (int j = 0; j < n_tgt; ++j) {
            const float* tgt = target_bounds + j * 4;
            int gx_min = std::max(0, (int)((tgt[0] - minx) / cell_w));
            int gx_max = std::min(grid_x - 1, (int)((tgt[2] - minx) / cell_w));
            int gy_min = std::max(0, (int)((tgt[1] - miny) / cell_h));
            int gy_max = std::min(grid_y - 1, (int)((tgt[3] - miny) / cell_h));

            std::unordered_map<int, bool> seen;

            for (int gx = gx_min; gx <= gx_max; ++gx) {
                for (int gy = gy_min; gy <= gy_max; ++gy) {
                    int key = grid_hash(gx, gy, grid_x);
                    auto it = grid.find(key);
                    if (it != grid.end()) {
                        for (int src_id : it->second) {
                            if (!seen[src_id]) {
                                seen[src_id] = true;
                                const float* src = source_bounds + src_id * 4;
                                if (bboxes_intersect(src, tgt)) {
                                    local_pairs.push_back({src_id, j});
                                }
                            }
                        }
                    }
                }
            }
        }

        #pragma omp critical
        all_pairs.insert(all_pairs.end(), local_pairs.begin(), local_pairs.end());
    }

    *out_count = all_pairs.size();
    IntPair* result = (IntPair*)malloc(sizeof(IntPair) * all_pairs.size());
    std::memcpy(result, all_pairs.data(), sizeof(IntPair) * all_pairs.size());
    return result;
}

void free_grid_pairs(IntPair* ptr) {
    free(ptr);
}

}

