# gnuplot
### 1. Set the separator to a comma
set datafile separator ","

### 2. Plot columns 1 (x) and 2 (y)
plot "data.csv" using 1:2 with lines
