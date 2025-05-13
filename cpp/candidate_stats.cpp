#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <omp.h>

extern "C" {

// Compute feature13 (sum of frequency) and feature14 (number of candidates)
// for each targetId in the input list, using CSR-format candidate lists.
void compute_candidate_stats(
    const int* target_ids, int N,
    const int* candidate_offsets, const int* candidate_values,
    const int* frequency,
    float* out_freq_sums,
    int* out_candidate_counts
) {
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        int tgt_id = target_ids[i];
        int start = candidate_offsets[tgt_id];
        int end = candidate_offsets[tgt_id + 1];

        float freq_sum = 0.0f;
        for (int j = start; j < end; ++j) {
            int src_id = candidate_values[j];
            freq_sum += frequency[src_id];
        }

        out_freq_sums[i] = freq_sum;
        out_candidate_counts[i] = end - start;
    }
}

}
