#ifndef SORT_H
#define SORT_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

void count_sort_init(uint8_t *bytes,
                     uint32_t *indices,
                     uint32_t size,
                     uint32_t *counts,
                     uint32_t *parts,
                     uint32_t *part_count);

void merge_sort_2(__m256i *values, __m256i *indices);

void merge_sort_4(__m256i *values, __m256i *indices);

void merge_sort_8(__m256i *values, __m256i *indices);

void merge_sort_16(__m256i *values, __m256i *indices);

#endif // SORT_H
