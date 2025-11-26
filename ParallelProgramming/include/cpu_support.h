#pragma once
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

uint32_t random_word() {
    uint32_t random;

    random = rand() & 0xFFFF;
    random = (random << 16) + (rand() & 0xFFFF);
    return random;
}

void random_words(uint32_t *x, size_t size) {
    for (size_t i = 0; i < size; i++) {
        x[i] = random_word();
    }
}

void print_words(const uint32_t *x, size_t size) {
    for (size_t i = 0; i < size; i++) {
        printf("%08x\n", x[i]);
    }
}

void zero_words(uint32_t *x, size_t size) {
    for (size_t i = 0; i < size; i++) {
        x[i] = 0;
    }
}

bool compare_results(const uint32_t *a, const uint32_t *b, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (a[i] != b[i]) {
            printf("Mismatch at index %zu: %08x != %08x\n", i, a[i], b[i]);
            return false;
        }
    }
}

double get_time_in_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

