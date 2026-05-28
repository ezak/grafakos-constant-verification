#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <mpfr.h>
#include <omp.h>
#include <vector>

enum class TargetFunction
{
  F_HK                 = 0,
  FF_HK_OPTIMIZED      = 1,
  F_SECH               = 2,
  F_GAUSSIAN           = 3,
  F_EXTREMAL           = 4,
  F_EXTREMAL_CANDIDATE = 5
};

constexpr mpfr_prec_t PRECISION = 256;

// RAII Container for managed arrays of mpfr_t
class MpfrArray
{
private:
  std::vector<mpfr_t> data;

public:
  MpfrArray (size_t size, mpfr_prec_t prec = PRECISION) : data (size)
  {
    for (auto &item : data)
      {
        mpfr_init2 (item, prec);
      }
  }

  ~MpfrArray ()
  {
    for (auto &item : data)
      {
        mpfr_clear (item);
      }
  }

  // Disable copying to prevent double-free bugs
  MpfrArray (const MpfrArray &)            = delete;
  MpfrArray &operator= (const MpfrArray &) = delete;

  mpfr_t &
  operator[] (size_t idx)
  {
    return data[idx];
  }
  const mpfr_t &
  operator[] (size_t idx) const
  {
    return data[idx];
  }
  mpfr_t *
  data_ptr ()
  {
    return data.data ();
  }
};

// Thread-local scratchpad to eliminate dynamic allocations inside hotspots
struct ThreadWorkspace
{
  mpfr_t tmp, result_fk, result_htk, x_j, sum_at_xj, arg, j, k;

  ThreadWorkspace (mpfr_prec_t prec = PRECISION)
  {
    mpfr_inits2 (prec, tmp, result_fk, result_htk, x_j, sum_at_xj, arg, j, k, nullptr);
  }
  ~ThreadWorkspace ()
  {
    mpfr_clears (tmp, result_fk, result_htk, x_j, sum_at_xj, arg, j, k, nullptr);
  }

  ThreadWorkspace (const ThreadWorkspace &)            = delete;
  ThreadWorkspace &operator= (const ThreadWorkspace &) = delete;
};

// Math kernels (Thread-safe)
void
f_hk (mpfr_t response, const mpfr_t h, const mpfr_t k)
{
  mpfr_mul (response, h, k, MPFR_RNDN);
  mpfr_sqr (response, response, MPFR_RNDN);
  mpfr_add_si (response, response, 1, MPFR_RNDN);
  mpfr_si_div (response, 1, response, MPFR_RNDN);
}

void
ff_hk_optimized (mpfr_t response, mpfr_t tmp, const mpfr_t h, const mpfr_t k, const mpfr_t p)
{
  mpfr_mul (response, h, k, MPFR_RNDN);
  int sgn = mpfr_sgn (response);
  mpfr_abs (response, response, MPFR_RNDN);

  mpfr_t epsilon;
  mpfr_init_set (epsilon, h, MPFR_RNDN);
  mpfr_mul_d (epsilon, epsilon, 0.1, MPFR_RNDN);

  if (mpfr_cmp (response, epsilon) < 0 || mpfr_cmp_d (response, 10000.0) > 0)
    {
      mpfr_set_zero (response, 1);
    }
  else
    {
      mpfr_si_div (tmp, -1, p, MPFR_RNDN);
      mpfr_pow (response, response, tmp, MPFR_RNDN);

      if (sgn < 0)
        mpfr_neg (response, response, MPFR_RNDN);
      else if (sgn == 0)
        mpfr_set_zero (response, 1);
    }
  mpfr_clear (epsilon);
}

void
f_sech (mpfr_t response, const mpfr_t h, const mpfr_t k)
{
  mpfr_mul (response, h, k, MPFR_RNDN);
  mpfr_cosh (response, response, MPFR_RNDN);
  mpfr_ui_div (response, 1, response, MPFR_RNDN);
}

void
f_gaussian (mpfr_t response, const mpfr_t h, const mpfr_t k)
{
  mpfr_mul (response, h, k, MPFR_RNDN);
  mpfr_sqr (response, response, MPFR_RNDN);
  mpfr_neg (response, response, MPFR_RNDN);
  mpfr_exp (response, response, MPFR_RNDN);
}

void
f_extremal_candidate (mpfr_t response, mpfr_t tmp, const mpfr_t h, const mpfr_t k, const mpfr_t p)
{
  mpfr_mul (tmp, h, k, MPFR_RNDN);
  int sgn = mpfr_sgn (tmp);
  mpfr_abs (tmp, tmp, MPFR_RNDN);

  if (mpfr_cmp_d (tmp, 1e-15) < 0 || mpfr_cmp_d (tmp, 1e7) > 0)
    {
      mpfr_set_zero (response, 1);
    }
  else
    {
      mpfr_si_div (response, -1, p, MPFR_RNDN);
      mpfr_pow (response, tmp, response, MPFR_RNDN);
      if (sgn < 0)
        mpfr_neg (response, response, MPFR_RNDN);
    }
}

void
f_extremal (mpfr_t result, mpfr_t tmp, const mpfr_t h, const mpfr_t k, const mpfr_t pi)
{
  mpfr_mul (tmp, h, k, MPFR_RNDN);
  mpfr_abs (result, tmp, MPFR_RNDN);
  mpfr_mul_ui (result, result, 2, MPFR_RNDN);

  mpfr_mul (tmp, tmp, tmp, MPFR_RNDN);
  mpfr_mul_ui (tmp, tmp, 4, MPFR_RNDN);
  mpfr_ui_sub (tmp, 1, tmp, MPFR_RNDN);

  if (mpfr_sgn (tmp) < 0)
    {
      mpfr_set_zero (result, 1);
      return;
    }

  mpfr_sqrt (tmp, tmp, MPFR_RNDN);
  mpfr_add_ui (tmp, tmp, 1, MPFR_RNDN);
  mpfr_div (tmp, tmp, result, MPFR_RNDN);
  mpfr_log (tmp, tmp, MPFR_RNDN);

  mpfr_ui_div (result, 2, pi, MPFR_RNDN);
  mpfr_mul (result, result, tmp, MPFR_RNDN);
}

void
hilbert_transform_kernel (mpfr_t result, mpfr_t tmp, const mpfr_t arg, const mpfr_t threshold, const mpfr_t pi)
{
  if (mpfr_cmpabs (arg, threshold) < 0)
    {
      mpfr_set_zero (result, 1);
    }
  else
    {
      mpfr_mul (tmp, pi, arg, MPFR_RNDN);
      mpfr_cos (result, tmp, MPFR_RNDN);
      mpfr_si_sub (result, 1, result, MPFR_RNDN);
      mpfr_div (result, result, tmp, MPFR_RNDN);
    }
}

// Routes evaluated calculations
void
dispatch_function (int fun, mpfr_t out, mpfr_t tmp, const mpfr_t h, const mpfr_t k, const mpfr_t p, const mpfr_t pi)
{
  switch (static_cast<TargetFunction> (fun))
    {
    case TargetFunction::F_HK:
      f_hk (out, h, k);
      break;
    case TargetFunction::FF_HK_OPTIMIZED:
      ff_hk_optimized (out, tmp, h, k, p);
      break;
    case TargetFunction::F_SECH:
      f_sech (out, h, k);
      break;
    case TargetFunction::F_GAUSSIAN:
      f_gaussian (out, h, k);
      break;
    case TargetFunction::F_EXTREMAL:
      f_extremal (out, tmp, h, k, pi);
      break;
    case TargetFunction::F_EXTREMAL_CANDIDATE:
      f_extremal_candidate (out, tmp, h, k, p);
      break;
    }
}

// Multi-threaded Source Sampling
void
sample_source (MpfrArray &result, const mpfr_t h, long N_long, const mpfr_t p, const mpfr_t pi, int fun)
{
#pragma omp parallel
  {
    mpfr_t local_k, local_tmp;
    mpfr_inits2 (PRECISION, local_k, local_tmp, nullptr);

#pragma omp for schedule(static)
    for (long k_val = -N_long; k_val <= N_long; ++k_val)
      {
        long index = k_val + N_long;
        mpfr_set_si (local_k, k_val, MPFR_RNDN);
        dispatch_function (fun, result[index], local_tmp, h, local_k, p, pi);
      }
    mpfr_clears (local_k, local_tmp, nullptr);
  }
}

// Highly Parallel O(N^2) Hilbert Transform Loop
void
sample_hilbert_transform (MpfrArray &result, const mpfr_t h, long N_long, const mpfr_t p, const mpfr_t threshold, const mpfr_t pi, int fun)
{
#pragma omp parallel
  {
    ThreadWorkspace ws;

#pragma omp for schedule(dynamic, 16)
    for (long j_val = -N_long; j_val <= N_long; ++j_val)
      {
        long index_j = j_val + N_long;
        mpfr_set_si (ws.j, j_val, MPFR_RNDN);
        mpfr_set_zero (ws.sum_at_xj, 1);

        for (long k_val = -N_long; k_val <= N_long; ++k_val)
          {
            if (j_val == k_val)
              continue;

            mpfr_set_si (ws.k, k_val, MPFR_RNDN);
            mpfr_sub (ws.arg, ws.j, ws.k, MPFR_RNDN);

            dispatch_function (fun, ws.result_fk, ws.tmp, h, ws.k, p, pi);
            hilbert_transform_kernel (ws.result_htk, ws.tmp, ws.arg, threshold, pi);

            mpfr_mul (ws.result_fk, ws.result_fk, ws.result_htk, MPFR_RNDN);
            mpfr_add (ws.sum_at_xj, ws.sum_at_xj, ws.result_fk, MPFR_RNDN);
          }
        mpfr_set (result[index_j], ws.sum_at_xj, MPFR_RNDN);
      }
  }
}

void
lp_norm (mpfr_t result, mpfr_t tmp, MpfrArray &samples, int size, const mpfr_t p, const mpfr_t h)
{
  mpfr_set_zero (result, 1);
  for (int i = 0; i < size; i++)
    {
      mpfr_abs (tmp, samples[i], MPFR_RNDN);
      mpfr_pow (tmp, tmp, p, MPFR_RNDN);
      mpfr_mul (tmp, tmp, h, MPFR_RNDN);
      mpfr_add (result, result, tmp, MPFR_RNDN);
    }
  mpfr_ui_div (tmp, 1, p, MPFR_RNDN);
  mpfr_pow (result, result, tmp, MPFR_RNDN);
}

void
compute_grafakos_constant (mpfr_t result, const mpfr_t p, const mpfr_t pi)
{
  mpfr_sub_d (result, p, 1, MPFR_RNDN);
  mpfr_div (result, p, result, MPFR_RNDN);

  if (mpfr_cmp (p, result) > 0)
    {
      mpfr_set (result, p, MPFR_RNDN);
    }

  mpfr_mul_d (result, result, 2.0, MPFR_RNDN);
  mpfr_div (result, pi, result, MPFR_RNDN);
  mpfr_tan (result, result, MPFR_RNDN);
  mpfr_d_div (result, 1.0, result, MPFR_RNDN);
}

int
main (int argc, char **argv)
{
  // 1. Thread Safety Check
  if (!mpfr_buildopt_tls_p ())
    {
      std::cerr << "Critical Error: MPFR dependency lacks thread-safety capabilities.\n";
      return 1;
    }

  float L   = 300;
  int   PP  = 3;
  int   MAX = 10;
  int   FUN = 0;

  int opt;
  int option_index = 0;
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

  mpfr_set_default_prec (PRECISION);

  // Context execution variables
  mpfr_t numerator, h, N, p, max, grafakos, pi;
  mpfr_t src_norm, hilb_norm, tmp, threshold, ratio, err;
  mpfr_inits2 (PRECISION, numerator, h, N, p, max, grafakos, pi, src_norm, hilb_norm, tmp, threshold, ratio, err, nullptr);

  mpfr_set_d (numerator, 1.0, MPFR_RNDN);
  mpfr_set_d (p, PP, MPFR_RNDN);
  mpfr_set_d (max, MAX, MPFR_RNDN);
  mpfr_set_d (threshold, 1e-12, MPFR_RNDN);
  mpfr_const_pi (pi, MPFR_RNDN);

  compute_grafakos_constant (grafakos, p, pi);

  const int array_size = 1024 * 1024;
  MpfrArray samples (array_size);
  MpfrArray HF_xi_samples (array_size);

  for (long i = 1; i < MAX; i++)
    {
      mpfr_div_ui (h, numerator, i, MPFR_RNDN);
      mpfr_d_div (N, L, h, MPFR_RNDN);

      long current_N_long       = mpfr_get_si (N, MPFR_RNDN);
      int  actual_samples_count = 2 * current_N_long + 1;

      if (actual_samples_count > array_size)
        {
          actual_samples_count = array_size;
        }

      // Run Parallel Calculations
      sample_source (samples, h, current_N_long, p, pi, FUN);
      lp_norm (src_norm, tmp, samples, actual_samples_count, p, h);

      sample_hilbert_transform (HF_xi_samples, h, current_N_long, p, threshold, pi, FUN);
      lp_norm (hilb_norm, tmp, HF_xi_samples, actual_samples_count, p, h);

      mpfr_div (ratio, hilb_norm, src_norm, MPFR_RNDN);
      mpfr_sub (err, grafakos, ratio, MPFR_RNDN);

      mpfr_printf ("%.6Rf %.60Rf  %.60Rf\n", h, ratio, grafakos);
    }

  mpfr_clears (numerator, h, N, p, max, grafakos, pi, src_norm, hilb_norm, tmp, threshold, ratio, err, nullptr);
  mpfr_free_cache ();

  return 0;
}