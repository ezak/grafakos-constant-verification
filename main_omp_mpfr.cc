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
  explicit MpfrArray (size_t size, mpfr_prec_t prec = PRECISION) : data (size)
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
};

// Thread-local scratchpad to eliminate dynamic allocations inside kernels
struct ThreadWorkspace
{
  mpfr_t tmp, result_fk, result_htk, x_j, sum_at_xj, arg, j, k;
  mpfr_t local_k, local_tmp;

  ThreadWorkspace (mpfr_prec_t prec = PRECISION)
  {
    mpfr_inits2 (prec, tmp, result_fk, result_htk, x_j, sum_at_xj, arg, j, k, local_k, local_tmp, nullptr);
  }
  ~ThreadWorkspace ()
  {
    mpfr_clears (tmp, result_fk, result_htk, x_j, sum_at_xj, arg, j, k, local_k, local_tmp, nullptr);
  }

  ThreadWorkspace (const ThreadWorkspace &)            = delete;
  ThreadWorkspace &operator= (const ThreadWorkspace &) = delete;
};

// Math kernels
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

// 1. Thread-Isolated Source Sampling
void
sample_source (MpfrArray &result, ThreadWorkspace &ws, const mpfr_t h, long N_long, const mpfr_t p, const mpfr_t pi, int fun)
{
  for (long k_val = -N_long; k_val <= N_long; ++k_val)
    {
      long index = k_val + N_long;
      mpfr_set_si (ws.local_k, k_val, MPFR_RNDN);
      dispatch_function (fun, result[index], ws.local_tmp, h, ws.local_k, p, pi);
    }
}

// 2. Thread-Isolated Hilbert Transform Loop
void
sample_hilbert_transform (MpfrArray &result, ThreadWorkspace &ws, const mpfr_t h, long N_long, const mpfr_t p, const mpfr_t threshold, const mpfr_t pi, int fun)
{
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
  if (!mpfr_buildopt_tls_p ())
    {
      std::cerr << "Critical Error: MPFR dependency lacks thread-safety capabilities.\n";
      return 1;
    }

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

  mpfr_set_default_prec (PRECISION);

  // Theoretical limit remains thread-shared read-only
  mpfr_t shared_p, shared_pi, shared_grafakos;
  mpfr_inits2 (PRECISION, shared_p, shared_pi, shared_grafakos, nullptr);
  mpfr_set_d (shared_p, PP, MPFR_RNDN);
  mpfr_const_pi (shared_pi, MPFR_RNDN);
  compute_grafakos_constant (shared_grafakos, shared_p, shared_pi);

  const int array_size = 1024 * 1024;

  // Output synchronization structure
  struct OutputData
  {
    double      h_val;
    std::string ratio_str;
    std::string grafakos_str;
  };
  std::vector<OutputData> results_buffer (MAX);

// HOISTED OPENMP BLOCK: Splitting the convergence iterations among CPU threads.
// The parallel region is kept plain to establish the thread pool safely.
#pragma omp parallel
  {
    // Every hardware core instantiates exactly ONE set of working memory for its lifetime
    ThreadWorkspace ws;
    MpfrArray       samples (array_size);
    MpfrArray       HF_xi_samples (array_size);

    mpfr_t numerator, h, N, src_norm, hilb_norm, tmp_norm, threshold, ratio;
    mpfr_inits2 (PRECISION, numerator, h, N, src_norm, hilb_norm, tmp_norm, threshold, ratio, nullptr);

    mpfr_set_d (numerator, 1.0, MPFR_RNDN);
    mpfr_set_d (threshold, 1e-12, MPFR_RNDN);

// FIX: The schedule clause is moved here onto the 'omp for' directive where it is valid.
// 'nowait' is kept to avoid an unnecessary secondary barrier before thread-local cleanup.
#pragma omp for schedule(dynamic, 1) nowait
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

        // Execute local operations safely within the thread's personal sandbox
        sample_source (samples, ws, h, current_N_long, shared_p, shared_pi, FUN);
        lp_norm (src_norm, tmp_norm, samples, actual_samples_count, shared_p, h);

        sample_hilbert_transform (HF_xi_samples, ws, h, current_N_long, shared_p, threshold, shared_pi, FUN);
        lp_norm (hilb_norm, tmp_norm, HF_xi_samples, actual_samples_count, shared_p, h);

        mpfr_div (ratio, hilb_norm, src_norm, MPFR_RNDN);

        // Print format conversion safely inside each thread without printing out-of-order
        char *r_buf = nullptr, *g_buf = nullptr;
        mpfr_asprintf (&r_buf, "%.60Rf", ratio);
        mpfr_asprintf (&g_buf, "%.60Rf", shared_grafakos);

        results_buffer[i] = { mpfr_get_d (h, MPFR_RNDN), std::string (r_buf), std::string (g_buf) };

        mpfr_free_str (r_buf);
        mpfr_free_str (g_buf);
      }

    mpfr_clears (numerator, h, N, src_norm, hilb_norm, tmp_norm, threshold, ratio, nullptr);
  } // OpenMP implicit join fence here. Threads clean up their buffers independently.

  // Sequential Print step: Guarantees your simulation data logs cleanly from i=1 to MAX
  for (long i = 1; i < MAX; i++)
    {
      if (!results_buffer[i].ratio_str.empty ())
        {
          std::printf ("%.6f %s  %s\n", results_buffer[i].h_val,
                       results_buffer[i].ratio_str.c_str (),
                       results_buffer[i].grafakos_str.c_str ());
        }
    }

  mpfr_clears (shared_p, shared_pi, shared_grafakos, nullptr);
  mpfr_free_cache ();
  return 0;
}