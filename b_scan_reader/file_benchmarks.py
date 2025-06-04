import BScan
import numpy as np
import random
import timeit

def benchmark_tell():
    for _ in range(1000000):
        b_scan.f.tell()

def benchmark_seek():
    for i in range(1000000):
        b_scan.f.seek(i)

def benchmark_repeated_seek():
    for _ in range(1000000):
        b_scan.f.seek(12345)

def benchmark_random_seek():
    for i in range(1000000):
        b_scan.f.seek(seek_positions[i])

def benchmark_read():
    for _ in range(1000000):
        _ = b_scan.f.read(2)


def benchmark_random_seek_and_read():
    for i in range(1000000):
        b_scan.f.seek(seek_positions[i])
        _ = b_scan.f.read(2)

### R-09
input_scan = "scans/BSCAN Type D  R-09 Pickering A Unit-1 east 30-Jan-2020 105718 [A2536-2556][R1380-1800].anf"


b_scan = BScan.BScan(input_scan)

channel = b_scan.get_channel_info('ID1 NB')

# Build up a random array of seek_positions outside of the function so that it doesn't affect the run time.
seek_positions = random.sample(range(0, 2000000), 1000000)

print("Running Benchmarks")
print("Reported times are how lonw it takes to call the function 1,000,000 times.")
print("benchmark_tell() - ", np.round(timeit.timeit("benchmark_tell()", setup="from __main__ import benchmark_tell", number=1),2),"s")
print("benchmark_seek() - ", np.round(timeit.timeit("benchmark_seek()", setup="from __main__ import benchmark_seek", number=1),2),"s")
print("benchmark_random_seek() - ", np.round(timeit.timeit("benchmark_random_seek()", setup="from __main__ import benchmark_random_seek", number=1),2),"s")
print("benchmark_repeated_seek() - ", np.round(timeit.timeit("benchmark_repeated_seek()", setup="from __main__ import benchmark_repeated_seek", number=1),2),"s")
b_scan.f.seek(4096)
print("benchmark_read() - ", np.round(timeit.timeit("benchmark_read()", setup="from __main__ import benchmark_read", number=1),2),"s")
print("benchmark_random_seek_and_read() - ", np.round(timeit.timeit("benchmark_random_seek_and_read()", setup="from __main__ import benchmark_random_seek_and_read", number=1),2),"s")