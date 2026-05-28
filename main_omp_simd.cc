#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

// Google Highway headers
#undef HWY_TARGET_TOGGLES
#define HWY_TARGET HWY_STATIC_TARGET
#include <format>
#include <hwy/contrib/math/math-inl.h>
#include <hwy/highway.h>

enum class TargetFunction
{
  F_HK                 = 0,
  FF_HK_OPTIMIZED      = 1,
  F_SECH               = 2,
  F_GAUSSIAN           = 3,
  F_EXTREMAL           = 4,
  F_EXTREMAL_CANDIDATE = 5
};

/**
 * @brief Calculates a two-dimensional Cauchy-Lorentzian profile based on the product of two coordinates.
 *
 * This function evaluates a symmetric, algebraic decay curve centered at @f$h \cdot k = 0@f$.
 * It treats the squared product of the input variables as the quadratic variance component,
 * producing a smooth, heavy-tailed regularization or low-pass windowing filter.
 *
 * Mathematically, the function evaluates to:
 * @f[
 * f(h, k) = \frac{1}{(hk)^2 + 1}
 * @f]
 *
 * @note **Mathematical & Function Space Characteristics:**
 *       - Unlike the exponential decay of a Gaussian profile (@c f_gaussian_scalar), this function
 *         exhibits **algebraic decay** (@f$\sim x^{-2}@f$) as @f$|h \cdot k| \to \infty@f$.
 *       - Because the tails drop off relatively slowly, its integrability over @f$\mathbb{R}^2@f$ depends
 *         heavily on the context of the domain. It is bounded, has no singularities, and belongs to
 *         higher @f$L^p@f$ spaces (such as @f$L^2@f$ and @f$L^3@f$).
 *
 * @param h       The primary coordinate component (e.g., Miller index or frequency dimension h).
 * @param k       The secondary coordinate component (e.g., Miller index or frequency dimension k).
 *
 * @return double The calculated filter weight in the open-closed interval @f$(0, 1.0]@f$.
 *                Returns @c 1.0 when @f$h = 0@f$ or @f$k = 0@f$, and approaches @c 0.0 asymptotically
 *                as the magnitude of the product @f$|h \cdot k|@f$ grows large.
 *
 * @see f_gaussian_scalar
 * @see f_sech_scalar
 */
double
f_hk_scalar (double h, double k)
{
  double hk = h * k;
  return 1.0 / (hk * hk + 1.0);
}

/**
 * @brief Calculates an optimized, signed power-law scaling factor with threshold filtering.
 *
 * This function computes a negative power-law decay curve based on the product of two
 * independent variables (typically frequency components or Miller indices @f$h@f$ and @f$k@f$).
 * It applies strict dynamic and static bounding boxes to prevent numerical instability,
 * division-by-zero errors, and floating-point overflows.
 *
 * Mathematically, the function evaluates to:
 * @f[
 * f(h, k, p) = \begin{cases}
 * 0 & \text{if } |hk| < 0.1h \text{ or } |hk| > 10000 \\
 * \text{sgn}(hk) \cdot |hk|^{-\frac{1}{p}} & \text{otherwise}
 * \end{cases}
 * @f]
 *
 * @note Because the function features strict upper and lower cutoffs, it possesses compact
 *       support and no singularities. Thus, it belongs to the @f$L^p@f$ function spaces (including @f$L^3@f$).
 *
 * @param h       The primary coordinate or scaling dimension (e.g., Miller index h).
 *                Determines the dynamic lower threshold value (@f$\epsilon = 0.1 \cdot h@f$).
 * @param k       The secondary coordinate or dimension (e.g., Miller index k).
 * @param p       The power-law shape parameter. Controls the steepness of the decay curve.
 *                Must not be zero to prevent division by zero in the exponent.
 *
 * @return double The signed scaling factor if parameters are within bounds; otherwise @c 0.0.
 *
 * @pre  The parameter @p p should be non-zero (@f$p \neq 0@f$).
 * @post The return value is guaranteed to be within the range @f$[-A, A]@f$ where
 *       @f$A = |0.1h|^{-\frac{1}{p}}@f$, mitigating any risk of producing @c NaN or @c Inf.
 */
double
ff_hk_optimized_scalar (double h, double k, double p)
{
  double hk      = h * k;
  double abs_hk  = std::abs (hk);
  double epsilon = h * 0.1;

  if (abs_hk < epsilon || abs_hk > 10000.0)
    {
      return 0.0;
    }
  else
    {
      double res = std::pow (abs_hk, -1.0 / p);
      if (hk < 0.0)
        return -res;
      if (hk == 0.0)
        return 0.0;
      return res;
    }
}

/**
 * @brief Calculates the hyperbolic secant (sech) of the product of two variables.
 *
 * This function computes the standard hyperbolic secant scaling factor, defined as the
 * reciprocal of the hyperbolic cosine. It creates a smooth, bell-shaped localization
 * window centered at @f$h \cdot k = 0@f$.
 *
 * Mathematically, the function evaluates to:
 * @f[
 * f(h, k) = \text{sech}(h \cdot k) = \frac{1}{\cosh(h \cdot k)} = \frac{2}{e^{h \cdot k} + e^{-h \cdot k}}
 * @f]
 *
 * @note Unlike raw power-law functions, this function is inherently bounded between @f$(0, 1]@f$
 *       and has no singularities along the real axis. It decays exponentially as @f$|h \cdot k| \to \infty@f$.
 *       Because its tails drop exponentially, this function is smoothly integrable and belongs to
 *       all @f$L^p@f$ spaces (including @f$L^1, L^2,@f$ and @f$L^3@f$) over @f$\mathbb{R}^2@f$.
 *
 * @param h       The primary coordinate component (e.g., Miller index or frequency dimension h).
 * @param k       The secondary coordinate component (e.g., Miller index or frequency dimension k).
 *
 * @return double The calculated scaling factor in the open-closed interval @f$(0, 1.0]@f$.
 *                Returns @c 1.0 when @f$h \cdot k = 0@f$, and approaches @c 0.0 as @f$|h \cdot k|@f$ grows large.
 *
 * @see ff_hk_optimized_scalar
 */
double
f_sech_scalar (double h, double k)
{
  return 1.0 / std::cosh (h * k);
}

/**
 * @brief Calculates a two-dimensional Gaussian profile based on the product of two coordinates.
 *
 * This function evaluates a symmetric, bell-shaped Gaussian distribution centered at
 * @f$h \cdot k = 0@f$. It uses the squared product of the input variables as its exponent,
 * creating a highly localized smoothing or windowing filter.
 *
 * Mathematically, the function evaluates to:
 * @f[
 * f(h, k) = e^{-(h \cdot k)^2}
 * @f]
 *
 * @note This function is inherently bounded within the interval @f$(0, 1.0]@f$ and is
 *       infinitely differentiable (smooth) across the entire real domain. Because it exhibits
 *       superexponential decay (@f$e^{-x^2}@f$) as @f$|h \cdot k| \to \infty@f$, it is rapidly
 *       integrable and belongs to the Schwartz space as well as all @f$L^p@f$ spaces (including @f$L^3@f$).
 *
 * @param h       The primary coordinate component (e.g., Miller index or frequency dimension h).
 * @param k       The secondary coordinate component (e.g., Miller index or frequency dimension k).
 *
 * @return double The calculated Gaussian weight in the open-closed interval @f$(0, 1.0]@f$.
 *                Returns @c 1.0 when @f$h = 0@f$ or @f$k = 0@f$, and approaches @c 0.0 rapidly
 *                as the product @f$|h \cdot k|@f$ increases.
 *
 * @see f_sech_scalar
 * @see ff_hk_optimized_scalar
 */
double
f_gaussian_scalar (double h, double k)
{
  double hk = h * k;
  return std::exp (-(hk * hk));
}


/**
 * @brief Evaluates an unwindowed, signed power-law decay curve with static hardware safety cutoffs.
 *
 * This function calculates a raw negative power-law scaling factor based on the product
 * of two variables (@f$h@f$ and @f$k@f$). Unlike functions that employ relative scaling windows,
 * this function relies entirely on fixed, hard-coded limits to safeguard against machine-level
 * floating-point exceptions (division-by-zero, underflow, and overflow).
 *
 * Mathematically, the function evaluates to:
 * @f[
 * f(h, k, p) = \begin{cases}
 * 0 & \text{if } |hk| < 10^{-15} \text{ or } |hk| > 10^7 \\
 * \text{sgn}(hk) \cdot |hk|^{-\frac{1}{p}} & \text{otherwise}
 * \end{cases}
 * @f]
 *
 * @note Because this function uses fixed static thresholds rather than dynamic or relative
 *       boundaries, its integration region forms a deterministic hyperbola cross-section on @f$\mathbb{R}^2@f$.
 *       The hard bounds ensure the function has compact support and no singularities, confirming
 *       its membership in all @f$L^p@f$ function spaces (including @f$L^3@f$).
 *
 * @param h       The primary coordinate component (e.g., Miller index or frequency dimension h).
 * @param k       The secondary coordinate component (e.g., Miller index or frequency dimension k).
 * @param p       The power-law shape parameter. Dictates the severity of the attenuation.
 *                Must not be zero to avoid division by zero in the exponent calculation.
 *
 * @return double The signed scaling factor if the product @f$|h \cdot k|@f$ falls within the stable range;
 *                otherwise returns @c 0.0.
 *
 * @pre  The parameter @p p should be non-zero (@f$p \neq 0@f$).
 *
 * @see ff_hk_optimized_scalar
 */
double
f_extremal_candidate_scalar (double h, double k, double p)
{
  double hk     = h * k;
  double abs_hk = std::abs (hk);

  if (abs_hk < 1e-15 || abs_hk > 1e7)
    {
      return 0.0;
    }
  double res = std::pow (abs_hk, -1.0 / p);
  return (hk < 0.0) ? -res : res;
}


/**
 * @brief Calculates a specialized logarithmic scaling factor with domain boundaries.
 *
 * This function evaluates an analytical expression involving a square root and a
 * logarithm based on the product of two coordinates (@f$h@f$ and @f$k@f$). It features
 * a strict upper bound on the coordinate product to guarantee real-valued square roots
 * and avoid complex numbers.
 *
 * Mathematically, the function evaluates to:
 * @f[
 * f(h, k, \pi) = \begin{cases}
 * 0 & \text{if } 1 - 4(hk)^2 < 0 \\
 * \frac{2}{\pi} \ln\left( \frac{\sqrt{1 - 4(hk)^2} + 1}{2|hk|} \right) & \text{otherwise}
 * \end{cases}
 * @f]
 *
 * @note **Mathematical Constraints & Domain:**
 *       - The radical argument @f$1 - 4(hk)^2 \ge 0@f$ restricts the valid input domain
 *         to the interval @f$|hk| \le 0.5@f$. Outside this range, the function returns @c 0.0.
 *       - As @f$|hk| \to 0@f$, the numerator inside the logarithm approaches @f$2.0@f$ while
 *         the denominator @f$2|hk| \to 0@f$. This creates a singularity where the function
 *         tends toward @f$+\infty@f$.
 *
 * @param h       The primary coordinate component (e.g., Miller index or frequency dimension h).
 * @param k       The secondary coordinate component (e.g., Miller index or frequency dimension k).
 * @param pi      The mathematical constant pi (@f$\pi \approx 3.141592653589793@f$).
 *                Must be non-zero to avoid division by zero.
 *
 * @return double The calculated scaling factor if the product @f$|h \cdot k| \le 0.5@f$;
 *                otherwise returns @c 0.0.
 *
 * @pre  The product @f$|h \cdot k|@f$ should be strictly greater than zero (@f$|hk| > 0@f$) to
 *       prevent an open division-by-zero or infinite logarithm result at runtime.
 *
 * @see f_extremal_candidate_scalar
 */
double
f_extremal_scalar (double h, double k, double pi)
{
  double hk        = h * k;
  double res_denom = std::abs (hk) * 2.0;

  double tmp = 1.0 - (hk * hk * 4.0);
  if (tmp < 0.0)
    return 0.0;

  double log_arg = (std::sqrt (tmp) + 1.0) / res_denom;
  return (2.0 / pi) * std::log (log_arg);
}


/**
 * @brief Evaluates the discrete kernel for a modified or truncated Hilbert transform.
 *
 * This function calculates the scalar weight of a 1D Hilbert transform kernel at a given
 * coordinate distance (@f$\text{arg}@f$). The Hilbert transform acts as a wideband
 * \(90^\circ\) phase shifter, often used in signal processing to construct analytic signals
 * or analyze single-sideband modulations.
 *
 * Mathematically, the function evaluates to:
 * @f[
 * f(t, \epsilon, \pi) = \begin{cases}
 * 0 & \text{if } |t| < \epsilon \\
 * \frac{1 - \cos(\pi t)}{\pi t} & \text{otherwise}
 * \end{cases}
 * @f]
 *
 * @note **Mathematical & Filter Characteristics:**
 *       - **Even Symmetry / Cancellation:** When @f$t@f$ is an even integer (e.g., \(\pm 2, \pm 4\)),
 *         \(\cos(\pi t) = 1\), causing the numerator to become zero. The kernel naturally vanishes
 *         at these points.
 *       - **Odd Peak Attenuation:** When @f$t@f$ is an odd integer (e.g., \(\pm 1, \pm 3\)),
 *         \(\cos(\pi t) = -1\), simplifying the non-zero envelope to @f$\frac{2}{\pi t}@f$.
 *
 * @param arg       The coordinate distance or sample offset from the center of the kernel (@f$t@f$).
 * @param threshold The safety floor radius (@f$\epsilon@f$). Prevents the @f$1/t@f$ singularity
 *                  by hard-clipping values near the center to @c 0.0.
 * @param pi        The mathematical constant pi (@f$\pi \approx 3.141592653589793@f$).
 *
 * @return double The calculated kernel weight if @f$|\text{arg}| \ge \text{threshold}@f$;
 *                otherwise returns @c 0.0.
 *
 * @pre  The @p threshold parameter should be strictly greater than zero (@f$\epsilon > 0@f$) to
 *       guarantee that the function never attempts a division by zero when @p arg is @c 0.0.
 */
double
hilbert_transform_kernel_scalar (double arg, double threshold, double pi)
{
  if (std::abs (arg) < threshold)
    {
      return 0.0;
    }
  double tmp = pi * arg;
  return (1.0 - std::cos (tmp)) / tmp;
}

double
dispatch_function_scalar (int fun, double h, double k, double p, double pi)
{
  switch (static_cast<TargetFunction> (fun))
    {
    case TargetFunction::F_HK:
      return f_hk_scalar (h, k);
    case TargetFunction::FF_HK_OPTIMIZED:
      return ff_hk_optimized_scalar (h, k, p);
    case TargetFunction::F_SECH:
      return f_sech_scalar (h, k);
    case TargetFunction::F_GAUSSIAN:
      return f_gaussian_scalar (h, k);
    case TargetFunction::F_EXTREMAL:
      return f_extremal_scalar (h, k, pi);
    case TargetFunction::F_EXTREMAL_CANDIDATE:
      return f_extremal_candidate_scalar (h, k, p);
    }
  return 0.0;
}

// ============================================================================
// HIGHWAY SIMD VECTORIZED KERNELS
// ============================================================================

HWY_BEFORE_NAMESPACE ();
namespace project
{
namespace HWY_NAMESPACE
{

namespace hn = hwy::HWY_NAMESPACE;

template <class D, class V>
V
f_hk_simd (D d, V h, V k)
{
  V hk  = hn::Mul (h, k);
  V hk2 = hn::Mul (hk, hk);
  V den = hn::Add (hk2, hn::Set (d, 1.0));
  return hn::Div (hn::Set (d, 1.0), den);
}

template <class D, class V>
V
ff_hk_optimized_simd (D d, V h, V k, V p)
{
  V hk      = hn::Mul (h, k);
  V abs_hk  = hn::Abs (hk);
  V epsilon = hn::Mul (h, hn::Set (d, 0.1));

  V zero      = hn::Zero (d);
  V minus_one = hn::Set (d, -1.0);
  V one       = hn::Set (d, 1.0);
  V sgn       = hn::IfThenElse (hn::Lt (hk, zero), minus_one, hn::IfThenElse (hn::Gt (hk, zero), one, zero));

  V inv_p    = hn::Div (hn::Set (d, -1.0), p);
  V pow_val  = hn::Exp (d, hn::Mul (inv_p, hn::Log (d, abs_hk)));
  V response = hn::Mul (pow_val, sgn);

  auto mask_zero = hn::Or (hn::Lt (abs_hk, epsilon), hn::Gt (abs_hk, hn::Set (d, 10000.0)));
  return hn::IfThenElse (mask_zero, zero, response);
}

template <class D, class V>
V
f_sech_simd (D d, V h, V k)
{
  V x = hn::Mul (h, k);
  // cosh(x) = (exp(x) + exp(-x)) / 2 -> sech(x) = 2 / (exp(x) + exp(-x))
  V exp_x     = hn::Exp (d, x);
  V exp_neg_x = hn::Exp (d, hn::Neg (x));
  V den       = hn::Add (exp_x, exp_neg_x);
  return hn::Div (hn::Set (d, 2.0), den);
}

template <class D, class V>
V
f_gaussian_simd (D d, V h, V k)
{
  V hk  = hn::Mul (h, k);
  V hk2 = hn::Mul (hk, hk);
  return hn::Exp (d, hn::Neg (hk2));
}

template <class D, class V>
V
f_extremal_candidate_simd (D d, V h, V k, V p)
{
  V hk        = hn::Mul (h, k);
  V abs_hk    = hn::Abs (hk);
  V zero      = hn::Zero (d);
  V minus_one = hn::Set (d, -1.0);
  V one       = hn::Set (d, 1.0);
  V sgn       = hn::IfThenElse (hn::Lt (hk, zero), minus_one, hn::IfThenElse (hn::Gt (hk, zero), one, zero));

  V inv_p    = hn::Div (hn::Set (d, -1.0), p);
  V pow_val  = hn::Exp (d, hn::Mul (inv_p, hn::Log (d, abs_hk)));
  V response = hn::Mul (pow_val, sgn);

  auto mask_zero = hn::Or (hn::Lt (abs_hk, hn::Set (d, 1e-15)), hn::Gt (abs_hk, hn::Set (d, 1e7)));
  return hn::IfThenElse (mask_zero, zero, response);
}

template <class D, class V>
V
f_extremal_simd (D d, V h, V k, V pi)
{
  V hk        = hn::Mul (h, k);
  V res_denom = hn::Mul (hn::Abs (hk), hn::Set (d, 2.0));

  V hk2 = hn::Mul (hk, hk);
  V tmp = hn::Sub (hn::Set (d, 1.0), hn::Mul (hk2, hn::Set (d, 4.0)));

  auto mask_neg = hn::Lt (tmp, hn::Zero (d));

  V sqrt_tmp = hn::Sqrt (tmp);
  V num      = hn::Add (sqrt_tmp, hn::Set (d, 1.0));
  V log_arg  = hn::Div (num, res_denom);
  V log_val  = hn::Log (d, log_arg);

  V factor = hn::Div (hn::Set (d, 2.0), pi);
  V result = hn::Mul (factor, log_val);

  return hn::IfThenElse (mask_neg, hn::Zero (d), result);
}

template <class D, class V>
V
hilbert_transform_kernel_simd (D d, V arg, V threshold, V pi)
{
  V tmp     = hn::Mul (pi, arg);
  V cos_val = hn::Cos (d, tmp);
  V num     = hn::Sub (hn::Set (d, 1.0), cos_val);
  V result  = hn::Div (num, tmp);

  auto mask_thresh = hn::Lt (hn::Abs (arg), threshold);
  return hn::IfThenElse (mask_thresh, hn::Zero (d), result);
}

template <class D, class V>
V
dispatch_function_simd (int fun, D d, V h, V k, V p, V pi)
{
  switch (static_cast<TargetFunction> (fun))
    {
    case TargetFunction::F_HK:
      return f_hk_simd (d, h, k);
    case TargetFunction::FF_HK_OPTIMIZED:
      return ff_hk_optimized_simd (d, h, k, p);
    case TargetFunction::F_SECH:
      return f_sech_simd (d, h, k);
    case TargetFunction::F_GAUSSIAN:
      return f_gaussian_simd (d, h, k);
    case TargetFunction::F_EXTREMAL:
      return f_extremal_simd (d, h, k, pi);
    case TargetFunction::F_EXTREMAL_CANDIDATE:
      return f_extremal_candidate_simd (d, h, k, p);
    }
  return hn::Zero (d);
}

// 1. SIMD Source Sampling
void
sample_source_simd (double *result, double h_val, long N_long, double p_val, double pi_val, int fun)
{
  const hn::ScalableTag<double> d;
  using V = hn::Vec<decltype (d)>;

  const size_t lanes          = hn::Lanes (d);
  long         total_elements = 2 * N_long + 1;

  V h  = hn::Set (d, h_val);
  V p  = hn::Set (d, p_val);
  V pi = hn::Set (d, pi_val);

  std::vector<double> init_k (lanes);
  for (size_t lane = 0; lane < lanes; ++lane)
    init_k[lane] = static_cast<double> (lane);
  V k_step = hn::Load (d, init_k.data ());

  long i = 0;
  for (; i <= total_elements - static_cast<long> (lanes); i += lanes)
    {
      long k_start_val = -N_long + i;
      V    k_base      = hn::Set (d, static_cast<double> (k_start_val));
      V    k           = hn::Add (k_base, k_step);

      V res = dispatch_function_simd (fun, d, h, k, p, pi);
      hn::Store (res, d, result + i);
    }

  // Scalar Tail Cleanup
  for (; i < total_elements; ++i)
    {
      long k_val = -N_long + i;
      result[i]  = dispatch_function_scalar (fun, h_val, static_cast<double> (k_val), p_val, pi_val);
    }
}

// 2. SIMD Hilbert Transform Loop Optimization
void
sample_hilbert_transform_simd (double *result, double h_val, long N_long, double p_val, double threshold_val, double pi_val, int fun)
{
  const hn::ScalableTag<double> d;
  const size_t                  lanes          = hn::Lanes (d);
  long                          total_elements = 2 * N_long + 1;

  using V = hn::Vec<decltype (d)>;

  V h         = hn::Set (d, h_val);
  V p         = hn::Set (d, p_val);
  V threshold = hn::Set (d, threshold_val);
  V pi        = hn::Set (d, pi_val);

  std::vector<double> init_k (lanes);
  for (size_t lane = 0; lane < lanes; ++lane)
    init_k[lane] = static_cast<double> (lane);
  V k_step = hn::Load (d, init_k.data ());

  for (long j_val = -N_long; j_val <= N_long; ++j_val)
    {
      long index_j = j_val + N_long;
      V    j_v     = hn::Set (d, static_cast<double> (j_val));
      V    sum_v   = hn::Zero (d);

      long k_idx = 0;
      for (; k_idx <= total_elements - static_cast<long> (lanes); k_idx += lanes)
        {
          long k_start_val = -N_long + k_idx;
          V    k_base      = hn::Set (d, static_cast<double> (k_start_val));
          V    k_v         = hn::Add (k_base, k_step);

          auto mask_not_equal = hn::Not (hn::Eq (j_v, k_v));
          V    arg            = hn::Sub (j_v, k_v);

          V fk  = dispatch_function_simd (fun, d, h, k_v, p, pi);
          V htk = hilbert_transform_kernel_simd (d, arg, threshold, pi);

          V prod = hn::Mul (fk, htk);
          sum_v  = hn::Add (sum_v, hn::IfThenElse (mask_not_equal, prod, hn::Zero (d)));
        }

      double total_sum = hn::ReduceSum (d, sum_v);

      // Scalar Tail Cleanup
      for (; k_idx < total_elements; ++k_idx)
        {
          long k_val = -N_long + k_idx;
          if (j_val == k_val)
            continue;

          double arg_scalar = static_cast<double> (j_val - k_val);
          double fk_scalar  = dispatch_function_scalar (fun, h_val, static_cast<double> (k_val), p_val, pi_val);
          double htk_scalar = hilbert_transform_kernel_scalar (arg_scalar, threshold_val, pi_val);

          total_sum += fk_scalar * htk_scalar;
        }

      result[index_j] = total_sum;
    }
}

// 3. SIMD Lp Norm Integration
double
lp_norm_simd (const double *samples, int size, double p_val, double h_val)
{
  const hn::ScalableTag<double> d;
  const size_t                  lanes = hn::Lanes (d);
  using V                             = hn::Vec<decltype (d)>;

  V p     = hn::Set (d, p_val);
  V h     = hn::Set (d, h_val);
  V sum_v = hn::Zero (d);

  int i = 0;
  for (; i <= size - static_cast<int> (lanes); i += lanes)
    {
      V sample_v = hn::Load (d, samples + i);
      V abs_v    = hn::Abs (sample_v);
      // x^p = exp(p * log(x))
      V pow_v = hn::Exp (d, hn::Mul (p, hn::Log (d, abs_v)));
      V term  = hn::Mul (pow_v, h);
      sum_v   = hn::Add (sum_v, term);
    }

  double total_sum = hn::ReduceSum (d, sum_v);

  // Scalar Tail Cleanup
  for (; i < size; ++i)
    {
      double abs_val = std::abs (samples[i]);
      total_sum += std::pow (abs_val, p_val) * h_val;
    }

  return std::pow (total_sum, 1.0 / p_val);
}

} // namespace HWY_NAMESPACE
} // namespace project
HWY_AFTER_NAMESPACE ();

// ============================================================================
// MAIN APPLICATION DRIVER
// ============================================================================

double
compute_grafakos_constant_scalar (double p, double pi)
{
  double result = p / (p - 1.0);
  if (p > result)
    {
      result = p;
    }

  result *= 2.0;
  return 1.0 / std::tan (pi / result);
}

int
main (int argc, char **argv)
{
  float L   = 300;
  int   PP  = 3;
  int   MAX = 10;
  int   FUN = 0;

  int opt, option_index = 0;
  int w_set = 0, p_set = 0;

  static struct option long_options[] = {
    { "window", required_argument, nullptr, 'w' },
    { "param", required_argument, nullptr, 'p' },
    { "loop", required_argument, nullptr, 'l' },
    { "function", required_argument, nullptr, 'f' },
    { nullptr, 0, nullptr, 0 }
  };

  while ((opt = getopt_long (argc, argv, "w:p:l:f:", long_options, &option_index)) != -1)
    {
      switch (opt)
        {
        case 'w':
          L     = std::atof (optarg);
          w_set = 1;
          break;
        case 'p':
          PP    = std::atoi (optarg);
          p_set = 1;
          break;
        case 'l':
          MAX = std::atoi (optarg);
          break;
        case 'f':
          FUN = std::atoi (optarg);
          break;
        default:
          std::cerr << "Usage: " << argv[0] << " --window <num> --param <num> [--loop <num>]\n";
          return EXIT_FAILURE;
        }
    }

  if (!w_set || !p_set)
    {
      std::cerr << "Error: Both --window and --param are required.\n";
      return EXIT_FAILURE;
    }

  const double shared_pi       = M_PI;
  const double shared_grafakos = compute_grafakos_constant_scalar (static_cast<double> (PP), shared_pi);
  const int    array_size      = 1024 * 1024;

  struct OutputData
  {
    double h_val;
    double ratio;
    double grafakos;
  };
  std::vector<OutputData> results_buffer (MAX);

// HOISTED OPENMP BLOCK
#pragma omp parallel
  {
    // Allocation-isolated memory structures instantiated per hardware thread
    std::vector<double> samples (array_size);
    std::vector<double> HF_xi_samples (array_size);

#pragma omp for schedule(dynamic, 1) nowait
    for (long i = 1; i < MAX; i++)
      {
        double h_val = 1.0 / static_cast<double> (i);
        double N_val = static_cast<double> (L) / h_val;

        long current_N_long       = static_cast<long> (N_val);
        int  actual_samples_count = 2 * current_N_long + 1;

        if (actual_samples_count > array_size)
          {
            actual_samples_count = array_size;
          }

        // SIMD Powered Kernels
        project::HWY_NAMESPACE::sample_source_simd (
            samples.data (), h_val, current_N_long, static_cast<double> (PP), shared_pi, FUN);

        double src_norm = project::HWY_NAMESPACE::lp_norm_simd (
            samples.data (), actual_samples_count, static_cast<double> (PP), h_val);

        project::HWY_NAMESPACE::sample_hilbert_transform_simd (
            HF_xi_samples.data (), h_val, current_N_long, static_cast<double> (PP), 1e-12, shared_pi, FUN);

        double hilb_norm = project::HWY_NAMESPACE::lp_norm_simd (
            HF_xi_samples.data (), actual_samples_count, static_cast<double> (PP), h_val);

        double ratio = hilb_norm / src_norm;

        results_buffer[i] = { h_val, ratio, shared_grafakos };
      }
  }

  std::string   csv_file = std::format ("out_p-{}_win-{}_func-{}_iter-{}_g-{}.csv", PP, L, FUN, MAX, shared_grafakos);
  std::ofstream file (csv_file);

  // Sequential Print step to guarantee sorted logs
  for (long i = 1; i < MAX; i++)
    {
      if (results_buffer[i].h_val > 0.0)
        {
          file << results_buffer[i].h_val << "," << results_buffer[i].ratio << "," << results_buffer[i].grafakos << std::endl;

          std::printf ("%.6f %.15f  %.15f\n",
                       results_buffer[i].h_val,
                       results_buffer[i].ratio,
                       results_buffer[i].grafakos);
        }
    }

  std::cout << "Done! " << std::endl;

  return 0;
}