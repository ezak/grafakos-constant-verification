# Step 1

### Install dependencies 
```shell
sudo apt install build-essential cmake ninja-build git libeigen3-dev libmpfr-dev libhwy-dev
```

# Step 2
### Build
```shell
cmake mkdir build && cd build && cmake -G Ninja -S ..
```

# Step 3
### copy this command from with in the build directory
```shell
(
echo "=== Starting Batch Run ==="

echo "--> Running Window 100"
{ time ./grafakos-constant-verification-main_omp_simd --window 100 --param 3 -f 1 -l 1000 ; } 2>&1 &&

echo "--> Running Window 200"
{ time ./grafakos-constant-verification-main_omp_simd --window 200 --param 3 -f 1 -l 2000 ; } 2>&1 &&

echo "--> Running Window 300"
{ time ./grafakos-constant-verification-main_omp_simd --window 300 --param 3 -f 1 -l 3000 ; } 2>&1 &&

echo "--> Running Window 400"
{ time ./grafakos-constant-verification-main_omp_simd --window 400 --param 3 -f 1 -l 4000 ; } 2>&1 &&

echo "--> Running Window 500"
{ time ./grafakos-constant-verification-main_omp_simd --window 500 --param 3 -f 1 -l 500 ; } 2>&1

echo "=== All Batches Finished ==="
) > output.log 2>&1 &
```

# Step 4

### Install gnuplot
```shell
sudo apt install gnuplot
```

### Run gnuplot from terminal using
```shell
gnuplot
```

### inside gnuplot shell execute to plot the convergence
```textmate
set datafile separator ","
plot "out_p-3_win-100_func-1_iter-100_g-1.7320508075688774.csv" using 1:3 with lines, \
"out_p-3_win-100_func-1_iter-100_g-1.7320508075688774.csv" using 1:2 with lines, \
"out_p-3_win-200_func-1_iter-100_g-1.7320508075688774.csv" using 1:2 with lines, \
"out_p-3_win-300_func-1_iter-100_g-1.7320508075688774.csv" using 1:2 with lines, \
"out_p-3_win-400_func-1_iter-100_g-1.7320508075688774.csv" using 1:2 with lines
```

### inside gnuplot shell execute to plot the error
```textmate
set datafile separator ","
plot "error_out_p-3_win-100_func-1_iter-100_g-1.7320508075688774.csv" using 1:4 with lines, \
"error_out_p-3_win-200_func-1_iter-100_g-1.7320508075688774.csv" using 1:4 with lines, \
"error_out_p-3_win-300_func-1_iter-100_g-1.7320508075688774.csv" using 1:4 with lines, \
"error_out_p-3_win-400_func-1_iter-100_g-1.7320508075688774.csv" using 1:4 with lines

```
