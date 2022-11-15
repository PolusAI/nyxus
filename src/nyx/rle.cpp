#include "rle.hpp"
#include <emmintrin.h>
#include <immintrin.h>

// Vector to shuffle 4-bit data into the lower 64 bits
const __m128i pshufbcnst_128 =
    _mm_set_epi8(0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x0E, 0x0C,
                 0x0A, 0x08, 0x06, 0x04, 0x02, 0x00);

// zero vector
const __m128i zeroes_128 = _mm_setzero_si128();
const __m256i zeroes_256 = _mm256_setzero_si256();

// Shuffle 4-bit vectors
const __m256i shf = _mm256_set_epi8(
    0x80, 0x0F, 0x80, 0x07, 0x80, 0x0E, 0x80, 0x06, 0x80, 0x0D, 0x80, 0x05,
    0x80, 0x0C, 0x80, 0x04, 0x80, 0x0B, 0x80, 0x03, 0x80, 0x0A, 0x80, 0x02,
    0x80, 0x09, 0x80, 0x01, 0x80, 0x08, 0x80, 0x00);

// Bit shifts to decode 4-bit packing
const __m256i shft = _mm256_set_epi64x(0x04, 0x00, 0x04, 0x00);

// For casting to 4-bit mask
const __m256i vmsk = _mm256_set1_epi8(0x0F);

// For casting 4-bit boolean to position index
const uint64_t cast_index = 0xFEDCBA9876543210;

/**
 * @brief Get indices of non-zero byte values in a 128 bit vector.
 *
 * Assuming data in a 128 bit vector is composed of byte data, this function
 * returns indices of non-zero values as packed uint16 values stored in a 256
 * bit vector.
 *
 * @param data
 * @return PackedIndex
 */
PackedIndex<__m256i> get_128_indices_word(__m128i data) {

  uint64_t ind; // Holds 4-bit indexes

  PackedIndex<__m256i> output{};

  // Determine how many edges were found
  data = _mm_srli_epi64(data, 4); // Pack 16x8 bit to 32x4 bit mask.
  data = _mm_shuffle_epi8(data, pshufbcnst_128); // Align the 16x4 bit mask.
  ind = ~_mm_cvtsi128_si64(data);                // Extract the 16x4 bit mask.
  output.count = static_cast<uint64_t>(_mm_popcnt_u64(ind)) >> 2u;
  ind = _pext_u64(cast_index, ind); // Get 1-15 index offset of edges

  // Unpack the 4 bit values
  output.indices = _mm256_set1_epi64x(ind);
  output.indices = _mm256_srlv_epi64(output.indices, shft);
  output.indices = _mm256_and_si256(output.indices, vmsk);
  output.indices = _mm256_shuffle_epi8(output.indices, shf);

  return output;
}

PackedIndex<__m256i> get_256_indices_word(__m256i data) {

  uint64_t ind; // Holds 4-bit indexes

  PackedIndex<__m256i> output{};

  // Determine how many edges were found
  ind = static_cast<unsigned int>(~_mm256_movemask_epi8(data));
  output.count = static_cast<uint64_t>(_mm_popcnt_u64(ind)) >> 2u;
  ind = _pext_u64(cast_index, ind); // Get 1-15 index offset of edges

  // Unpack the 4 bit values
  output.indices = _mm256_set1_epi64x(ind);
  output.indices = _mm256_srlv_epi64(output.indices, shft);
  output.indices = _mm256_and_si256(output.indices, vmsk);
  output.indices = _mm256_shuffle_epi8(output.indices, shf);

  return output;
}

void get_long_edges_256(const uint32_t *data, __m256i *mask, int i,
                        uint32_t last_val) {

  __m256i offset[4];

  int index = i * 8;
  int i1 = i + 1;
  int i2 = i + 2;
  int i3 = i + 3;

  // Unrolled data loading
  mask[i] = _mm256_loadu_si256((__m256i *)&data[index]);
  mask[i1] = _mm256_loadu_si256((__m256i *)&data[index + 8]);
  mask[i2] = _mm256_loadu_si256((__m256i *)&data[index + 16]);
  mask[i3] = _mm256_loadu_si256((__m256i *)&data[index + 24]);

  // Create shifted values
  offset[0] = _mm256_srli_si256(mask[i], 4);
  offset[1] = _mm256_srli_si256(mask[i1], 4);
  offset[2] = _mm256_srli_si256(mask[i2], 4);
  offset[3] = _mm256_srli_si256(mask[i3], 4);
  offset[0] = _mm256_insert_epi32(offset[0], _mm256_extract_epi32(mask[i1], 0), 7);
  offset[1] = _mm256_insert_epi32(offset[1], _mm256_extract_epi32(mask[i2], 0), 7);
  offset[2] = _mm256_insert_epi32(offset[2], _mm256_extract_epi32(mask[i3], 0), 7);
  offset[3] = _mm256_insert_epi32(offset[3], last_val, 7);
  offset[0] = _mm256_insert_epi32(offset[0], _mm256_extract_epi32(mask[i], 4), 3);
  offset[1] = _mm256_insert_epi32(offset[1], _mm256_extract_epi32(mask[i1], 4), 3);
  offset[2] = _mm256_insert_epi32(offset[2], _mm256_extract_epi32(mask[i2], 4), 3);
  offset[3] = _mm256_insert_epi32(offset[3], _mm256_extract_epi32(mask[i3], 4), 3);
  
  // Unrolled edge detection using xor
  mask[i] = _mm256_xor_si256(mask[i], offset[0]);
  mask[i1] = _mm256_xor_si256(mask[i1], offset[1]);
  mask[i2] = _mm256_xor_si256(mask[i2], offset[2]);
  mask[i3] = _mm256_xor_si256(mask[i3], offset[3]);

  // Generate 16x8 bit mask.
  mask[i] = _mm256_cmpeq_epi32(mask[i], zeroes_256);
  mask[i1] = _mm256_cmpeq_epi32(mask[i1], zeroes_256);
  mask[i2] = _mm256_cmpeq_epi32(mask[i2], zeroes_256);
  mask[i3] = _mm256_cmpeq_epi32(mask[i3], zeroes_256);

}

void get_byte_edges_128(const uint8_t *data, __m128i *mask, int i,
                        uint8_t last_val) {

  __m128i offset[4];

  int index = i * 16;
  int i1 = i + 1;
  int i2 = i + 2;
  int i3 = i + 3;

  // Unrolled data loading
  mask[i] = _mm_load_si128((__m128i *)&data[index]);
  mask[i1] = _mm_load_si128((__m128i *)&data[index + 16]);
  mask[i2] = _mm_load_si128((__m128i *)&data[index + 32]);
  mask[i3] = _mm_load_si128((__m128i *)&data[index + 48]);

  // Create shifted values
  offset[0] = _mm_bsrli_si128(mask[i], 1);
  offset[1] = _mm_bsrli_si128(mask[i1], 1);
  offset[2] = _mm_bsrli_si128(mask[i2], 1);
  offset[3] = _mm_bsrli_si128(mask[i3], 1);
  offset[0] = _mm_insert_epi8(offset[0], _mm_extract_epi8(mask[i1], 0), 15);
  offset[1] = _mm_insert_epi8(offset[1], _mm_extract_epi8(mask[i2], 0), 15);
  offset[2] = _mm_insert_epi8(offset[2], _mm_extract_epi8(mask[i3], 0), 15);
  offset[3] = _mm_insert_epi8(offset[3], last_val, 15);

  // Unrolled edge detection using xor
  mask[i] = _mm_xor_si128(mask[i], offset[0]);
  mask[i1] = _mm_xor_si128(mask[i1], offset[1]);
  mask[i2] = _mm_xor_si128(mask[i2], offset[2]);
  mask[i3] = _mm_xor_si128(mask[i3], offset[3]);

  // Generate 16x8 bit mask.
  mask[i] = _mm_cmpeq_epi8(mask[i], zeroes_128);
  mask[i1] = _mm_cmpeq_epi8(mask[i1], zeroes_128);
  mask[i2] = _mm_cmpeq_epi8(mask[i2], zeroes_128);
  mask[i3] = _mm_cmpeq_epi8(mask[i3], zeroes_128);
}

RLEStream<uint32_t, uint16_t> rle_encode_long_stream_32(const uint32_t *data,
                                                        const uint16_t len) {
  if ((len % 32) != 0) {
    throw std::invalid_argument(
        "rle.rle_encode_long_stream_32: len must be a multiple of 32. "
        "Arbitrary lengths should use rle_encode_stream.");
  }

  RLEStream<uint32_t, uint16_t> stream; // Initialize the output
  stream.offsets.resize(len + 1);       // Allocate memory
  uint16_t stream_ind = 0;              // Current index of stream.data
  __m256i edges;                        // Edge data

  stream.start = data;

  // Load the data and get offsets
  int num_chunks = len / 8;
  __m256i mask[num_chunks];

  // Loop over data
  for (int i = 0; i < num_chunks - 4; i += 4) {
    get_long_edges_256(data, mask, i, data[(i + 4) * 8]);
  }

  // Get edges of the last 16 values
  get_long_edges_256(data, mask, (num_chunks - 4), data[len - 1]);

  if (data[0] != 0u) {
    stream.offsets[stream_ind++] = 0;
  }

  for (int i = 0; i < num_chunks; ++i) {
    // Skip if no edges present
    if (_mm256_movemask_epi8(mask[i]) == 0xFFFFFFFF) {
      continue;
    }

    // Get the indices of edges
    PackedIndex<__m256i> edges = get_256_indices_word(mask[i]);
    edges.indices =
        _mm256_add_epi16(_mm256_set1_epi16(i * 8 + 1), edges.indices);

    // store the results
    _mm256_storeu_si256(
        reinterpret_cast<__m256i *>(&stream.offsets[stream_ind]),
        edges.indices);
    stream_ind += edges.count;
  }

  // Check the last value and add the index if present
  if (data[len - 1] != 0u) {
    stream.offsets[stream_ind++] = len;
  }

  stream.offsets.resize(stream_ind);

  return stream;
};

RLEStream<uint8_t, uint16_t> rle_encode_byte_stream_64(const uint8_t *data,
                                                       const uint16_t len) {
  if ((len % 64) != 0) {
    throw std::invalid_argument(
        "rle.rle_encode_byte_64: len must be a multiple of 64. "
        "Arbitrary lengths should use rle_encode_stream.");
  }

  RLEStream<uint8_t, uint16_t> stream; // Initialize the output
  stream.offsets.resize(len + 1);      // Allocate memory
  uint16_t stream_ind = 0;             // Current index of stream.data
  __m256i edges;                       // Edge data

  stream.start = data;

  // Load the data and get offsets
  int num_chunks = len / 16;
  __m128i mask[num_chunks];
  // Loop over data
  for (int i = 0; i < num_chunks - 4; i += 4) {
    get_byte_edges_128(data, mask, i, data[(i + 4) * 16]);
  }
  // Get edges of the late 64 bytes
  get_byte_edges_128(data, mask, (num_chunks - 4), data[len - 1]);

  if (data[0] != 0u) {
    stream.offsets[stream_ind++] = 0;
  }

  for (int i = 0; i < num_chunks; ++i) {
    // Skip if no edges present
    if (_mm_movemask_epi8(mask[i]) == 0xFFFFFFFFFFFFFFFF) {
      continue;
    }

    // Get the indices of edges
    PackedIndex<__m256i> edges = get_128_indices_word(mask[i]);
    edges.indices =
        _mm256_add_epi16(_mm256_set1_epi16(i * 16 + 1), edges.indices);

    // store the results
    _mm256_storeu_si256(
        reinterpret_cast<__m256i *>(&stream.offsets[stream_ind]),
        edges.indices);
    stream_ind += edges.count;
  }

  // Check the last value and add the index if present
  if (data[len - 1] != 0u) {
    stream.offsets[stream_ind++] = len;
  }

  stream.offsets.resize(stream_ind);

  return stream;
};

std::vector<RLEStream<uint8_t, uint16_t>>
rle_encode_byte_stream_64_tile(const uint8_t *data, const uint8_t *end,
                               const uint16_t stride,
                               const uint16_t chunk_size) {
  std::vector<RLEStream<uint8_t, uint16_t>> tile;

  for (; data < end; data += stride) {
    tile.push_back(rle_encode_byte_stream_64(data, chunk_size));
  }

  return tile;
}
