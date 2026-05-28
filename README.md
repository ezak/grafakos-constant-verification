# gnuplot
### 1. Set the separator to a comma
set datafile separator ","

### 2. Plot columns 1 (x) and 2 (y)
plot "data.csv" using 1:2 with lines


# Build

### Platform Dependencies
sudo apt install build-essential cmake ninja-build git

### Build Dependencies
sudo apt install libeigen3-dev libmpfr-dev libhwy-dev
