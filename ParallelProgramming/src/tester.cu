#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <algorithm>

#include "gpu_support.h"
#include "cpu_support.h"



int main(int argc, char* argv[]) {

    size_t n = size_t(1) << 20;

    int count = 1;
    if (argc > 1) count = atoi(argv[1]);

    std::vector<uint32_t> h(n);
    random_words(h.data(), n);

    double sort_cpu_time = get_time_in_seconds();
    sort(h.begin(), h.end());
    double elapsed_cpu_time = get_time_in_seconds() - sort_cpu_time;

    printf("CPU sort time: %.5f seconds\n", elapsed_cpu_time);

    return 0;
}   