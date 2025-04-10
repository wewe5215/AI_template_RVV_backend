#ifndef CUSTOM_MATH_CPU_H
#define CUSTOM_MATH_CPU_H

#include <cmath>
#include <cstdint>
#include <limits>

// For simplicity, we use float as our half type.
typedef _Float16 half;

// A simple 2-element vector for half precision.
// (Both components are stored as floats.)
struct half2 {
    _Float16 x;
    _Float16 y;
    
    // Default constructor sets both to 0.
    half2() : x(0.0f), y(0.0f) {}
    
    // Constructor from two values.
    half2(half a, half b) : x(a), y(b) {}

    // Constructor from a single value (broadcast to both components).
    explicit half2(half a) : x(a), y(a) {}
};

// Define elementwise operators for half2.
inline half2 operator+(const half2 &a, const half2 &b) {
    return half2(a.x + b.x, a.y + b.y);
}

inline half2 operator-(const half2 &a, const half2 &b) {
    return half2(a.x - b.x, a.y - b.y);
}

inline half2 operator*(const half2 &a, const half2 &b) {
    return half2(a.x * b.x, a.y * b.y);
}

inline half2 operator/(const half2 &a, const half2 &b) {
    return half2(a.x / b.x, a.y / b.y);
}


inline half2 __hmul2(const half2 &a, const half2 &b) { return a * b; }
inline half2 __hadd2(const half2 &a, const half2 &b) { return a + b; }
inline half2 __hsub2(const half2 &a, const half2 &b) { return a - b; }
inline half2 __h2div(const half2 &a, const half2 &b) { return a / b; }


// For single half arithmetic.
inline half __hmul(half a, half b) { return a * b; }
inline half __hadd(half a, half b) { return a + b; }
inline half __hsub(half a, half b) { return a - b; }
inline half __hdiv(half a, half b) { return a / b; }

inline float __hmul2_f32(const float a, const float b) { return a * b; }
inline float __hadd2_f32(const float a, const float b) { return a + b; }
inline float __hsub2_f32(const float a, const float b) { return a - b; }
inline float __h2div_f32(const float a, const float b) { return a / b; }

// Exponential functions.
inline _Float16 hexp(_Float16 a) { return std::exp((float)a); }
inline half2 h2exp(const half2 &a) { return half2(std::exp((float)(a.x)), std::exp((float)(a.y))); }

// Basic math functions
template <typename T>
inline T sign_custom(const T a) {
  return T(a > T(0)) - T(a < T(0));
}

inline half2 h2sign_custom(const half2 a) {
  return half2(sign_custom(a.x), sign_custom(a.y));
}

// Fast tanh approximations for half2 and half.
inline half2 fast_tanh(half2 x) {
  // 1 - 2/(e^(2x)+1)
  half2 u = __hmul2(half2(2.0f), x);
  half2 emu = h2exp(u);
  half2 cdf = __hsub2(half2(1.0f), __h2div(half2(2.0f), __hadd2(half2(1.0f), emu)));
  return cdf;
}

inline half fast_tanh(half x) {
  half u = __hmul(half(2.0f), x);
  half emu = hexp(u);
  half cdf = __hsub(half(1.0f), __hdiv(half(2.0f), __hadd(half(1.0f), emu)));
  return cdf;
}

// Returns 1 (for half, we simply return 1.0f)
inline float one() {
  return 1.0f;
}

/// Returns (1/2)
inline float constant_half() {
  return 0.5f;
}

inline float fsigmoid_custom(const float a) {
  return (std::tanh(a * 0.5f) + 1.0f) * 0.5f;
}

inline float hsigmoid_custom(const half a) {
  half half_val = constant_half();
  half one_val = one();
  return __hmul(__hadd(fast_tanh(__hmul(a, half_val)), one_val), half_val);
}

inline half2 h2sigmoid_custom(const half2 a) {
  half2 halfX2 = half2(constant_half());
  half2 oneX2  = half2(one());
  return __hmul2(__hadd2(fast_tanh(__hmul2(a, halfX2)), oneX2), halfX2);
}

inline float fsilu(const float a) {
  return a * fsigmoid_custom(a);
}

inline half hsilu(const half a) {
  return __hmul(a, hsigmoid_custom(a));
}

inline half2 h2silu(const half2 a) {
  return __hmul2(a, h2sigmoid_custom(a));
}

inline half hsin_custom(const half a) {
  // Since half is a float, we can simply use sin.
  return static_cast<half>(std::sin((float)a));
}

inline float leaky_relu(const float a, const float negativeSlope) {
  return a > 0.f ? a : a * negativeSlope;
}

inline half leaky_relu_f16(const float a, const half negativeSlope) {
  return a > 0.0f ? a : __hmul(a, negativeSlope);
}


inline float relu(const float a) {
  return a > 0.f ? a : 0.f;
}

inline half relu_f16(const half a) {
  return a > 0.0f ? a : 0.0f;
}

template <typename T>
inline T hard_tanh(const T a, T min_val, T max_val) {
  if (a <= min_val) {
    return min_val;
  } else if (a >= max_val) {
    return max_val;
  } else {
    return a;
  }
}

inline half2 h2hard_tanh(const half2 a, const half2 min_val, const half2 max_val) {
  return half2(hard_tanh(a.x, min_val.x, max_val.x),
               hard_tanh(a.y, min_val.y, max_val.y));
}

inline half replace_if_inf(const half a,
                             const half inf_replace,
                             const half neginf_replace) {
  if (std::isinf(a)) {
    return a < 0 ? neginf_replace : inf_replace;
  }
  return a;
}

inline float replace_if_inf(const float a,
                              const float inf_replace,
                              const float neginf_replace) {
  if (std::isinf(a)) {
    return a < 0 ? neginf_replace : inf_replace;
  }
  return a;
}

inline half2 nan_to_num(const half2 a,
                        const half2 nan_replace,
                        const half2 inf_replace,
                        const half2 neginf_replace) {
  return half2(std::isnan(a.x) ? nan_replace.x : replace_if_inf(a.x, inf_replace.x, neginf_replace.x),
               std::isnan(a.y) ? nan_replace.y : replace_if_inf(a.y, inf_replace.y, neginf_replace.y));
}

inline half nan_to_num_f16(const half a,
                       const half nan_replace,
                       const half inf_replace,
                       const half neginf_replace) {
  return std::isnan(a) ? nan_replace : replace_if_inf(a, inf_replace, neginf_replace);
}

inline half2 clamp_nan_to_num(const half2 a,
                              const half2 clamp_min,
                              const half2 clamp_max,
                              const half2 nan_replace) {
  return half2(std::isnan(a.x) ? nan_replace.x : hard_tanh(a.x, clamp_min.x, clamp_max.x),
               std::isnan(a.y) ? nan_replace.y : hard_tanh(a.y, clamp_min.y, clamp_max.y));
}

inline half clamp_nan_to_num_f16(const half a,
                             const half clamp_min,
                             const half clamp_max,
                             const half nan_replace) {
  return std::isnan(a) ? nan_replace : hard_tanh(a, clamp_min, clamp_max);
}

// Backup functions
inline half nanh() {
  return std::numeric_limits<float>::quiet_NaN();
}

inline bool half_isnan(half h) {
  return std::isnan(h);
}

inline half hmin(half a, half b) {
  return (a < b) ? a : b;
}

inline half hmax(half a, half b) {
  return (a > b) ? a : b;
}

inline float fmaxf_nan(const float a, const float b) {
  return (std::isnan(a) || std::isnan(b))
             ? std::numeric_limits<float>::quiet_NaN()
             : std::fmax(a, b);
}

inline half hmax_nan_f16(const half a, const half b) {
  return (half_isnan(a) || half_isnan(b)) ? nanh() : hmax(a, b);
}

inline float fminf_nan(const float a, const float b) {
  return (std::isnan(a) || std::isnan(b))
             ? std::numeric_limits<float>::quiet_NaN()
             : std::fmin(a, b);
}

inline half hmin_nan_f16(const half a, const half b) {
  return (half_isnan(a) || half_isnan(b)) ? nanh() : hmin(a, b);
}

#endif // CUSTOM_MATH_CPU_H
