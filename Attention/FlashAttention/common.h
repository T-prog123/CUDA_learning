#pragma once
#include <vector>
#include <tuple>

using MatrixTuple = std::tuple<std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>>;

MatrixTuple matrix_init(const int N, const int d) {
    int total_size = N * d;
    std::vector<float> h_Q(total_size);
    std::vector<float> h_K(total_size);
    std::vector<float> h_V(total_size);
    std::vector<float> h_O(total_size);

    for (int i = 0; i < total_size; ++i) {
        h_Q[i] = i % 20;
        h_K[i] = (i % N) + (i % d);
        h_V[i] = (3 * i + 5) % 16;
    }

    return {h_Q, h_K, h_V, h_O};
}