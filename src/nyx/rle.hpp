#ifndef _FTL_RLE_HPP
#define _FTL_RLE_HPP
/**
 * @file rle.hpp
 * @author Nick Schaub (nicholas.j.schaub@gmail.com)
 * @brief Optimized methods for performing run length encoding
 * @version 0.1
 * @date 2022-04-02
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <iostream>
#include <x86intrin.h>

/**
 * @brief Offsets from a a starting point where data changes value.
 *
 * Run Length Encoding (RLE) compresses data by finding consecutive,
 * repeated values. This is usually represented by a value followed by the
 * number of times the value should be repeated. This structure only records the
 * offsets to a change in value, and not the actual values themselves.
 *
 * A good explanation of RLE is in the packbits compression algorithm
 * Wikipedia page.
 *
 * NOTE: The quirk to this data representation is that the first offset may not
 * be 0, and in that case the values from start to the first offset are 0.
 *
 * https://en.wikipedia.org/wiki/PackBits
 *
 * @tparam T Data type of data to be compressed
 * @tparam U Data type of offsets to change in value
 */
template <typename T, typename U> struct RLEStream {
  const T *start;
  std::vector<U> offsets;
};

/**
 * @brief Run Length Encoding representation
 *
 * This structure contains all information needed for run length encoding. It
 * inherits from RLEStream, but unlike RLEStream it stores values for each
 * offset. Refer to RLEStream for additional details.
 *
 * @tparam T Data type of data to be compressed
 * @tparam U Data type of offsets to change in value
 */
template <typename T, typename U> struct RLE : public RLEStream<U, T> {
  std::vector<uint32_t> values;
};

// Structure to hold packed index
template <typename T> struct PackedIndex {
  static_assert(std::is_same<T, __m128i>::value or
#ifdef __AVX512BW__
                    std::is_same<T, __m512i>::value or
#endif
                    std::is_same<T, __m256i>::value,
                "!");
  T indices;
  uint8_t count;
};

RLEStream<uint8_t, uint16_t> rle_encode_byte_stream_64(const uint8_t *data,
                                                       uint16_t len);

RLEStream<uint32_t, uint16_t> rle_encode_long_stream_32(const uint32_t *data,
                                                        uint16_t len);

template <typename T>
RLEStream<T, uint16_t> rle_encode_byte_stream_64(const T *data, uint16_t len) {
  static_assert(std::is_same<T, unsigned char>::value or
                    std::is_same<T, char>::value or
                    std::is_same<T, bool>::value,
                "!");

  RLEStream<uint8_t, uint16_t> rle =
      rle_encode_byte_stream_64((uint8_t *)data, len);

  RLEStream<T, uint16_t> output = {(std::vector<T>)rle.offsets, rle.start};

  return output;
}

/**
 * @brief Universal RLE compressor, get offsets to changes in value.
 *
 * This function is a universal handler for calculating offsets to changes in
 * value. Handles all data stream types and streams of any size.
 *
 * @tparam U Data type of input data
 * @tparam T Data type of offsets
 * @param data Pointer to data for compression
 * @param len Length of stream to compress
 * @return RLEByteStream<U, T>
 */
template <typename U, typename T>
RLEStream<U, T> rle_encode_stream(const U *data, T len) {
  RLEStream<U, T> stream;
  stream.start = data;
  U on_object = 0;
  const U *end = data + len;
  const U *start = data;
  for (; data < end; data++) {
    if (*data) {
      if (*data != on_object) {
        stream.offsets.push_back((T)(data - start));
        on_object = *data;
      }
    } else if (on_object) {
      stream.offsets.push_back((T)(data - start));
      on_object = 0;
    }
  }

  if (on_object) {
    stream.offsets.push_back((T)(end - start));
  }

  return stream;
};

template <typename U, typename T>
std::vector<RLEStream<U, T>> rle_encode_stream_tile(const U *data, const U *end,
                                                    const T stride,
                                                    const T chunk_size) {
  std::vector<RLEStream<U, T>> tile;

  for (; data < end; data += stride) {
    tile.push_back(rle_encode_stream(data, chunk_size));
  }

  return tile;
}

std::vector<RLEStream<uint8_t, uint16_t>>
rle_encode_byte_stream_64_tile(const uint8_t *data, const uint8_t *end,
                               uint16_t stride, uint16_t chunk_size);

#endif // _FTL_RLE_HPP
