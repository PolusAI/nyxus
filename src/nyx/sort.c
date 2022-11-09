/*
 * Sorting algorithms for compression
 */

#include "sort.h"

/*
 * Sorting networks
 *
 * Below is a series of bitonic sorting network components. Sorting networks
 * for 2, 4, and 8 values receive all receive a 256-bit vector of 2, 4, or 8
 * uint32 values (respectively).
 *
 * NOTE: These sort data into descending order.
 *
 * TODO: Optimize the layered sorting to remove extraneous shuffle operations.
 *       For example, when sorting an array of 4 values merge_sort_2 is called
 *       which currently swaps the values into place if they are out of order.
 *       This could be optimized to swap values into the proper order for the
 *       next set of comparisons, thus removing one swap operation for each
 *       level of sorting.
 *
 * Ideas in these sorting implementations were taken from the following sources:
 * https://ieeexplore.ieee.org/document/8638394
 * https://www.vldb.org/pvldb/vol1/1454171.pdf
 */

// Swap adjacent integers into order
const uint8_t merge_swap_2 = 0xB1;
const uint8_t merge_shuf_2 = 0xF5;
void merge_sort_2(__m256i *values, __m256i *indices) {

    // Switch adjacent values
    __m256i values_swapped = _mm256_shuffle_epi32(*values,merge_swap_2);

    // Swap adjacent indices
    __m256i indices_swapped = _mm256_shuffle_epi32(*indices,merge_swap_2);

    // Compare values
    __m256i cmp = _mm256_cmpgt_epi32(*values,values_swapped);

    // Get blend indices
    cmp = _mm256_shuffle_epi32(cmp,merge_shuf_2);

    // Swap values
    *values = _mm256_blendv_epi8(*values,values_swapped,cmp);
    *indices = _mm256_blendv_epi8(*indices,indices_swapped,cmp);

}

// Sort integers in upper and lower lanes
const uint8_t merge_swap_4 = 0x1B;
const uint8_t merge_shuf_4 = 0xEB;
void merge_sort_4(__m256i *values, __m256i *indices) {

    // Sort adjacent integers
    merge_sort_2(values,indices);

    // Swap inner and outer values for comparison
    __m256i values_swapped = _mm256_shuffle_epi32(*values,merge_swap_4);

    // Swap inner and outer indices
    __m256i indices_swapped = _mm256_shuffle_epi32(*indices,merge_swap_4);

    // Compare values
    __m256i cmp = _mm256_cmpgt_epi32(*values,values_swapped);

    // Get blend indices
    cmp = _mm256_shuffle_epi32(cmp,merge_shuf_4);

    // Swap values
    *values = _mm256_blendv_epi8(*values,values_swapped,cmp);
    *indices = _mm256_blendv_epi8(*indices,indices_swapped,cmp);

    // Finally, sort adjacent integers
    merge_sort_2(values,indices);

}

// Sort integers in upper and lower lanes using interleaved comparisons
// Used when more than 4 values are compared and merge_sort_4 has been called
const uint8_t merge_swap_inter_4 = 0x4E;
const uint8_t merge_shuf_inter_4 = 0xEE;
void merge_sort_inter_4(__m256i *values, __m256i *indices) {

    // Swap inner and outer values for comparison
    __m256i values_swapped = _mm256_shuffle_epi32(*values,merge_swap_inter_4);

    // Swap inner and outer indices
    __m256i indices_swapped = _mm256_shuffle_epi32(*indices,merge_swap_inter_4);

    // Compare values
    __m256i cmp = _mm256_cmpgt_epi32(*values,values_swapped);

    // Get blend indices
    cmp = _mm256_shuffle_epi32(cmp,merge_shuf_inter_4);

    // Swap values
    *values = _mm256_blendv_epi8(*values,values_swapped,cmp);
    *indices = _mm256_blendv_epi8(*indices,indices_swapped,cmp);

    // Finally, sort adjacent integers
    merge_sort_2(values,indices);

}

// Sort integers across 128 bit lanes
void merge_sort_8(__m256i *values, __m256i *indices) {
    const __m256i merge_swap_8 = _mm256_set_epi32(0,1,2,3,4,5,6,7);
    const __m256i merge_shuf_8 = _mm256_set_epi32(7,6,5,4,4,5,6,7);

    // Sort integers within each lane
    merge_sort_4(values,indices);

    // Swap inner and outer values for comparison
    __m256i values_swapped = _mm256_permutevar8x32_epi32(*values,merge_swap_8);

    // Swap inner and outer indices
    __m256i indices_swapped = _mm256_permutevar8x32_epi32(*indices,merge_swap_8);

    // Compare values
    __m256i cmp = _mm256_cmpgt_epi32(*values,values_swapped);

    // Get blend indices
    cmp = _mm256_permutevar8x32_epi32(cmp,merge_shuf_8);

    // Swap values
    *values = _mm256_blendv_epi8(*values,values_swapped,cmp);
    *indices = _mm256_blendv_epi8(*indices,indices_swapped,cmp);

    // Values are sorted into high and low values, 
    merge_sort_inter_4(values,indices);

}

// Sort integers across 128 bit lanes using interleaved comparisons
// Used when more than 8 values are compared
void merge_sort_inter_8(__m256i *values, __m256i *indices) {
    const __m256i merge_swap_8 = _mm256_set_epi32(3,2,1,0,7,6,5,4);
    const __m256i merge_shuf_8 = _mm256_set_epi32(7,6,5,4,7,6,5,4);

    // Swap inner and outer values for comparison
    __m256i values_swapped = _mm256_permutevar8x32_epi32(*values,merge_swap_8);

    // Swap inner and outer indices
    __m256i indices_swapped = _mm256_permutevar8x32_epi32(*indices,merge_swap_8);

    // Compare values
    __m256i cmp = _mm256_cmpgt_epi32(*values,values_swapped);

    // Get blend indices
    cmp = _mm256_permutevar8x32_epi32(cmp,merge_shuf_8);

    // Swap values
    *values = _mm256_blendv_epi8(*values,values_swapped,cmp);
    *indices = _mm256_blendv_epi8(*indices,indices_swapped,cmp);

    // Values are sorted into high and low values, 
    merge_sort_inter_4(values,indices);

}

// Sort integers across vectors
// values and indices must be an array of __m256i vectors, and the first two
// are sorted
void merge_sort_16(__m256i *values, __m256i *indices) {
    const __m256i merge_swap_16 = _mm256_set_epi32(0,1,2,3,4,5,6,7);

    // Sort integers within each vector
    merge_sort_8(&values[0],&indices[0]);
    merge_sort_8(&values[1],&indices[1]);

    // Swap inner and outer values for comparison
    __m256i values_swapped[2];
    __m256i indices_swapped[2];
    values_swapped[0] = _mm256_permutevar8x32_epi32(values[0],merge_swap_16);
    indices_swapped[1] = _mm256_permutevar8x32_epi32(indices[1],merge_swap_16);
    values_swapped[1] = _mm256_permutevar8x32_epi32(values[1],merge_swap_16);
    indices_swapped[0] = _mm256_permutevar8x32_epi32(indices[0],merge_swap_16);

    // Compare values
    __m256i cmp = _mm256_cmpgt_epi32(values_swapped[1],values[0]);

    // Blend the first vector
    values[0] = _mm256_blendv_epi8(values[0],values_swapped[1],cmp);
    __m256i cmp_reversed = _mm256_permutevar8x32_epi32(cmp,merge_swap_16);
    indices[0] = _mm256_blendv_epi8(indices[0],indices_swapped[1],cmp);

    // Blend the second vector
    values[1] = _mm256_blendv_epi8(values[1],values_swapped[0],cmp_reversed);
    indices[1] = _mm256_blendv_epi8(indices[1],indices_swapped[0],cmp_reversed);

    // Values are sorted into high and low values, 
    merge_sort_inter_8(&values[0],&indices[0]);
    merge_sort_inter_8(&values[1],&indices[1]);

}

// Sort integers across vectors using interleaved comparisons
void merge_sort_inter_16(__m256i *values, __m256i *indices) {

    // Swap inner and outer values for comparison
    __m256i values_swapped[2];
    __m256i indices_swapped[2];
    values_swapped[0] = values[0];
    indices_swapped[1] = indices[1];
    values_swapped[1] = values[1];
    indices_swapped[0] = indices[0];

    // Compare values
    __m256i cmp = _mm256_cmpgt_epi32(values_swapped[1],values[0]);

    // Blend the first vector
    values[0] = _mm256_blendv_epi8(values[0],values_swapped[1],cmp);
    __m256i cmp_reversed = ~cmp;
    indices[0] = _mm256_blendv_epi8(indices[0],indices_swapped[1],cmp);

    // Blend the second vector
    values[1] = _mm256_blendv_epi8(values[1],values_swapped[0],cmp_reversed);
    indices[1] = _mm256_blendv_epi8(indices[1],indices_swapped[0],cmp_reversed);

    // Values are sorted into high and low values, 
    merge_sort_inter_8(&values[0],&indices[0]);
    merge_sort_inter_8(&values[1],&indices[1]);

}

// Sort integers across 4 vectors
// values and indices must be an array of __m256i vectors, and the first four
// are sorted
void merge_sort_16(__m256i *values, __m256i *indices) {
    const __m256i merge_swap_16 = _mm256_set_epi32(0,1,2,3,4,5,6,7);

    // Sort integers within each vector
    merge_sort_16(values,indices);
    merge_sort_16(values+2,indices+2);

    // Swap inner and outer values for comparison
    __m256i values_swapped[4];
    __m256i indices_swapped[4];
    values_swapped[0] = _mm256_permutevar8x32_epi32(values[0],merge_swap_16);
    indices_swapped[1] = _mm256_permutevar8x32_epi32(indices[1],merge_swap_16);
    values_swapped[1] = _mm256_permutevar8x32_epi32(values[1],merge_swap_16);
    indices_swapped[0] = _mm256_permutevar8x32_epi32(indices[0],merge_swap_16);

    // Compare values
    __m256i cmp[2];
    __m256i cmp[0] = _mm256_cmpgt_epi32(values_swapped[1],values[0]);

    // Blend the first vector
    values[0] = _mm256_blendv_epi8(values[0],values_swapped[1],cmp);
    __m256i cmp_reversed = _mm256_permutevar8x32_epi32(cmp,merge_swap_16);
    indices[0] = _mm256_blendv_epi8(indices[0],indices_swapped[1],cmp);

    // Blend the second vector
    values[1] = _mm256_blendv_epi8(values[1],values_swapped[0],cmp_reversed);
    indices[1] = _mm256_blendv_epi8(indices[1],indices_swapped[0],cmp_reversed);

    // Values are sorted into high and low values, 
    merge_sort_inter_8(&values[0],&indices[0]);
    merge_sort_inter_8(&values[1],&indices[1]);

}

/*
 * First pass binning algorithm for matching uint8 values
 *
 * This function determines the number of values in each bin of a histogram and
 * determines the offsets where values belong as a preliminary step in a count
 * sort algorithm. Each bin in the histogram represents a pair of bytes, so a
 * bin value of 10 in bin 0x0000, then there are exactly 10 locations in the
 * byte array where there is repeated null byte (with possible overlap).
 *
 * There are two steps to this algorithm
 * 1. Count the number of times each pair of bytes occurs.
 * 2. Calculate the offsets for where an index of a byte pair should be placed
 *    using count sort, such that the pairs of bytes that occur most frequently
 *    occur first.
 *
 * Modified from https://github.com/powturbo/Turbo-Histogram/blob/master/turbohist_.c
 */
static void bin_turbo_init(uint8_t *bytes,
                           uint32_t size,
                           uint32_t *counts,
                           uint32_t *parts,
                           uint32_t *part_count) {

    /*
     * Step 1: Count the number of occurences of each pair of bytes
     */

    // Two arrays to hold byte pair counts. Two arrays helps prevent collisions.
    uint32_t c[2][65536]={0};

    // Pointer to position in byte array
    uint8_t *ip;

    // Variable for preloading next set of bytes
    __m128i cpv = _mm_loadu_si128((__m128i*)bytes);

    // TODO: Remove the vector reversal as an optimization
    // Vector used to reverse the vector, useful when debugging.
    __m128i reverse = _mm_set_epi8( 0, 1, 2, 3, 4, 5, 6, 7,
                                    8, 9,10,11,12,13,14,15);

    // Loop through the data 32 bytes at a time.
    // Intrinsics are used to load the data to help reduce cache misses
    for( ip = bytes; ip != bytes+(size&~(32-1)); ) {
        ip+=16;

        // Load and reverse the vectors
        uint8_t cvt = ip[0];
        __m128i cv=cpv;
        __m128i dv = _mm_loadu_si128((__m128i*)(ip));
        cv = _mm_shuffle_epi8(cv,reverse);
        dv = _mm_shuffle_epi8(dv,reverse);
        ip+=16;
        uint8_t dvt = ip[0];
        cpv = _mm_loadu_si128((__m128i*)(ip));

        // Bin the data as uint16, bins even indexed values
        c[0][_mm_extract_epi16(cv,  0)]++;
        c[1][_mm_extract_epi16(dv,  0)]++;
        c[0][_mm_extract_epi16(cv,  1)]++;
        c[1][_mm_extract_epi16(dv,  1)]++;
        c[0][_mm_extract_epi16(cv,  2)]++;
        c[1][_mm_extract_epi16(dv,  2)]++;
        c[0][_mm_extract_epi16(cv,  3)]++;
        c[1][_mm_extract_epi16(dv,  3)]++;
        c[0][_mm_extract_epi16(cv,  4)]++;
        c[1][_mm_extract_epi16(dv,  4)]++;
        c[0][_mm_extract_epi16(cv,  5)]++;
        c[1][_mm_extract_epi16(dv,  5)]++;
        c[0][_mm_extract_epi16(cv,  6)]++;
        c[1][_mm_extract_epi16(dv,  6)]++;
        c[0][_mm_extract_epi16(cv,  7)]++;
        c[1][_mm_extract_epi16(dv,  7)]++;

        // shift, add in missing value, bin odd indexed values
        cv = _mm_slli_si128(cv,1);
        dv = _mm_slli_si128(dv,1);
        cv = _mm_insert_epi8(cv,cvt,0);
        dv = _mm_insert_epi8(dv,dvt,0);

        c[0][_mm_extract_epi16(cv,  0)]++;
        c[1][_mm_extract_epi16(dv,  0)]++;
        c[0][_mm_extract_epi16(cv,  1)]++;
        c[1][_mm_extract_epi16(dv,  1)]++;
        c[0][_mm_extract_epi16(cv,  2)]++;
        c[1][_mm_extract_epi16(dv,  2)]++;
        c[0][_mm_extract_epi16(cv,  3)]++;
        c[1][_mm_extract_epi16(dv,  3)]++;
        c[0][_mm_extract_epi16(cv,  4)]++;
        c[1][_mm_extract_epi16(dv,  4)]++;
        c[0][_mm_extract_epi16(cv,  5)]++;
        c[1][_mm_extract_epi16(dv,  5)]++;
        c[0][_mm_extract_epi16(cv,  6)]++;
        c[1][_mm_extract_epi16(dv,  6)]++;
        c[0][_mm_extract_epi16(cv,  7)]++;
        c[1][_mm_extract_epi16(dv,  7)]++;

        // Prefetch data to improve vector loading
        __builtin_prefetch(ip+512, 0);
    }

    // Finish binning the data
    uint16_t val;
    while( ip < bytes+size ) {
        val = *((uint16_t *) ip);
        c[0][(uint16_t) ((val << 8) + (val >> 8))]++;
        ip++;
    }

    /*
     * Step 2: Calculate the index offsets such that the indices of the most
     *         frequently occuring values occur first.
     */

    uint32_t indices[65536];
    __m256i index = _mm256_set_epi32(7,6,5,4,3,2,1,0);
    __m256i delta = _mm256_set_epi32(8,8,8,8,8,8,8,8);
    __m256i c0;
    for ( uint32_t i = 0; i < 65536; ) {
        _mm256_storeu_si256((__m256i *) &indices[i],index);
        c0 = _mm256_loadu_si256((__m256i *) &c[0][i]);
        __m256i c1 = _mm256_loadu_si256((__m256i *) &c[1][i]);
        index = _mm256_add_epi32(index,delta);
        c0 = _mm256_add_epi32(c0,c1);
        _mm256_storeu_si256((__m256i *) &c[0][i],c0);
        i += 8;
    }

    // Calculate the offsets - Could probably be optimized
    // Modified from https://stackoverflow.com/a/31229764
    int cmp( const void *a, const void *b ){
        int ia = *(int *)a;
        int ib = *(int *)b;
        return (c[0][ia] > c[0][ib]) - (c[0][ia] < c[0][ib]);
    }

    qsort(indices, 65536, sizeof(*indices), cmp);

    uint32_t count = 0;

    // Loop backwards so smallest clusters are last
    parts[0] = 0;
    (*part_count)++;
    for (int32_t i = 65535; i >= 0;--i) {
        counts[indices[i]] = count;
        count += c[0][indices[i]];
        if (count - counts[indices[i]] > 1) {
            parts[65536 - i] = count;
            (*part_count)++;
        }
    }
}

/*
 * Function for the first pass of sorting
 *
 * The first pass of sorting is simple: Count the number of times each pair of
 * characters  appears, then directly assign an index to the offset for each
 * set of values. This is a fast, stable sort. Subsequent sorting of partitions
 * requires slightly more complexity because previous sorting positions need to
 * be known, requiring a reference from a separate array.
 */
void count_sort_init(uint8_t *bytes,
                     uint32_t *indices,
                     uint32_t size,
                     uint32_t *counts,
                     uint32_t *parts,
                     uint32_t *part_count) {

    // Bin the data
    bin_turbo_init(bytes,size,counts,parts,part_count);
    uint32_t c;

    __m128i reverse = _mm_set_epi8( 0, 1, 2, 3, 4, 5, 6, 7,
                                    8, 9,10,11,12,13,14,15);

    // Sort the indices, unroll 16 times
    uint8_t *ip;
    __m256i index1 = _mm256_set_epi32( 7 ,6, 5, 4, 3, 2, 1, 0);
    __m256i index2 = _mm256_set_epi32(15,14,13,12,11,10, 9, 8);
    __m128i cpv = _mm_loadu_si128((__m128i*)bytes);
    __m256i delta = _mm256_set1_epi32(16);
    for(ip = bytes; ip != bytes+(size&~(16-1)); ) {
        // Load the values and start loading a new set of values
        __m128i cv = cpv;
        __m128i dv = _mm_srli_si128(cpv,1);
        ip += 16;
        cv = _mm_shuffle_epi8(cv,reverse);
        dv = _mm_insert_epi8(dv,ip[0],15);
        cpv = _mm_loadu_si128((__m128i*)(ip));
        dv = _mm_shuffle_epi8(dv,reverse);

        // Set the data
        indices[counts[_mm_extract_epi16(cv,  7)]++] = _mm256_extract_epi32(index1,  0);
        indices[counts[_mm_extract_epi16(dv,  7)]++] = _mm256_extract_epi32(index1,  1);
        indices[counts[_mm_extract_epi16(cv,  6)]++] = _mm256_extract_epi32(index1,  2);
        indices[counts[_mm_extract_epi16(dv,  6)]++] = _mm256_extract_epi32(index1,  3);
        indices[counts[_mm_extract_epi16(cv,  5)]++] = _mm256_extract_epi32(index1,  4);
        indices[counts[_mm_extract_epi16(dv,  5)]++] = _mm256_extract_epi32(index1,  5);
        indices[counts[_mm_extract_epi16(cv,  4)]++] = _mm256_extract_epi32(index1,  6);
        indices[counts[_mm_extract_epi16(dv,  4)]++] = _mm256_extract_epi32(index1,  7);
        index1 = _mm256_add_epi32(index1,delta);
        indices[counts[_mm_extract_epi16(cv,  3)]++] = _mm256_extract_epi32(index2,  0);
        indices[counts[_mm_extract_epi16(dv,  3)]++] = _mm256_extract_epi32(index2,  1);
        indices[counts[_mm_extract_epi16(cv,  2)]++] = _mm256_extract_epi32(index2,  2);
        indices[counts[_mm_extract_epi16(dv,  2)]++] = _mm256_extract_epi32(index2,  3);
        indices[counts[_mm_extract_epi16(cv,  1)]++] = _mm256_extract_epi32(index2,  4);
        indices[counts[_mm_extract_epi16(dv,  1)]++] = _mm256_extract_epi32(index2,  5);
        indices[counts[_mm_extract_epi16(cv,  0)]++] = _mm256_extract_epi32(index2,  6);
        indices[counts[_mm_extract_epi16(dv,  0)]++] = _mm256_extract_epi32(index2,  7);
        index2 = _mm256_add_epi32(index2,delta);
        __builtin_prefetch(ip+512, 0);
    }
    c = _mm256_extract_epi32(index1,  0);
    uint16_t val;
    while(ip < bytes+size) {
        val = *((uint16_t *) ip);
        indices[counts[(uint16_t) ((val << 8) + (val >> 8))]++] = c;
        c++;
        ip++;
    }
};
