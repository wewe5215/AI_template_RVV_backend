#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "macros.h"
#include "model_interface.h"
#include "raii_wrapper.h"

using namespace ait;

template <typename T>
static void make_random_integer_values(
    std::mt19937& rnd_generator,
    T* h_data,
    size_t numel,
    T lb,
    T ub) {
  std::uniform_int_distribution<> dist(lb, ub);
  for (size_t i = 0; i < numel; i++) {
    h_data[i] = static_cast<T>(dist(rnd_generator));
  }
}

static void make_random_float_values(
    std::mt19937& rnd_generator,
    float* h_data,
    size_t numel,
    float lb,
    float ub) {
  std::uniform_real_distribution<> dist(lb, ub);
  for (size_t i = 0; i < numel; i++) {
    h_data[i] = static_cast<float>(dist(rnd_generator));
  }
}

static void make_random_float16_values(
    std::mt19937& rnd_generator,
    _Float16* h_data,
    size_t numel,
    float lb,
    float ub) {
  std::uniform_real_distribution<> dist(lb, ub);
  for (size_t i = 0; i < numel; i++) {
    float v = static_cast<float>(dist(rnd_generator));
    // fixme: if encountered precision loss, then change the type conversion method
    h_data[i] = (_Float16)(v);
  }
}

static void make_random_bfloat16_values(
    std::mt19937& rnd_generator,
    bfloat16* h_data,
    size_t numel,
    float lb,
    float ub) {
  std::uniform_real_distribution<> dist(lb, ub);
  for (size_t i = 0; i < numel; i++) {
    float v = static_cast<float>(dist(rnd_generator));
    // fixme: if encountered precision loss, then change the type conversion method
    h_data[i] = (uint16_t)(v);
  }
}

static Ptr make_random_data(
    std::mt19937& rnd_generator,
    const AITemplateParamShape& shape,
    const AITemplateDtype& dtype) {
  size_t numel = shape.Numel();
  size_t num_bytes = numel * AITemplateDtypeSizeBytes(dtype);
  Ptr h_data = RAII_DeviceMalloc(num_bytes);
  switch (dtype) {
    case AITemplateDtype::kInt:
      make_random_integer_values<int>(
          rnd_generator,
          static_cast<int*>(h_data.get()),
          numel,
          /*lb*/ -10,
          /*ub*/ 10);
      break;
    case AITemplateDtype::kLong:
      make_random_integer_values<int64_t>(
          rnd_generator,
          static_cast<int64_t*>(h_data.get()),
          numel,
          /*lb*/ -10,
          /*ub*/ 10);
      break;
    case AITemplateDtype::kFloat:
      make_random_float_values(
          rnd_generator,
          static_cast<float*>(h_data.get()),
          numel,
          /*lb*/ 1.0,
          /*ub*/ 2.0);
      break;
    case AITemplateDtype::kBFloat16:
      make_random_bfloat16_values(
          rnd_generator,
          static_cast<uint16_t*>(h_data.get()),
          numel,
          /*lb*/ 1.0,
          /*ub*/ 2.0);
      break;
    case AITemplateDtype::kHalf:
      make_random_float16_values(
          rnd_generator,
          static_cast<_Float16*>(h_data.get()),
          numel,
          /*lb*/ 1.0,
          /*ub*/ 2.0);
      break;
    case AITemplateDtype::kBool:
      make_random_integer_values<bool>(
          rnd_generator,
          static_cast<bool*>(h_data.get()),
          numel,
          /*lb*/ 0,
          /*ub*/ 1);
      break;
    default:
      throw std::runtime_error("unsupported dtype for making random data");
  }

  return h_data;
}
