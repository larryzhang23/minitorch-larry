import timeit 
import warnings

import numba
from numba.core.errors import NumbaPerformanceWarning

import minitorch
from run_fast_tensor import datasets, FastTensorBackend, FastTrain

# Suppress Numba performance warnings
warnings.simplefilter("ignore", NumbaPerformanceWarning)
if numba.cuda.is_available():
    GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)

if __name__ == "__main__":
    PTS = 500
    HIDDEN = 1000
    data = datasets["Circle"](PTS)
    run_times = 5
    FastTrain(HIDDEN, backend=FastTensorBackend).train(data, 0.005, max_epochs=1)
    print("cpu", timeit.timeit("FastTrain(HIDDEN, backend=FastTensorBackend).train(data, 0.05, max_epochs=10)", number=run_times, globals=globals()) / run_times / 10)
    if numba.cuda.is_available():
        FastTrain(HIDDEN, backend=GPUBackend).train(data, 0.005, max_epochs=1)
        print("gpu", timeit.timeit("FastTrain(HIDDEN, backend=GPUBackend).train(data, 0.05, max_epochs=10)", number=run_times, globals=globals()) / run_times / 10)


