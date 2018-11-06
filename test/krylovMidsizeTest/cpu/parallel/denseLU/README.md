# Performance Test for CPU implementation of Krylov Subspace Method

This directory contains sample CPU implemented Krylov Subspace Method codes for performance testing.
Currently only CPU versions are incorporated into STACCATO build environment. For GPU implementations please refer to `STACCATO/test/krylovMidsizeTest/gpu`

## CPU Implementations

For execution, please run the binary (`./STACCATO_PerformanceTest`) to get information regarding command line arguments.

Currently the following combinations are finalised:

| Functional Parallelism | Dense LU | PARDISO |
|---|---|---|
|Sequential| yes | yes |
|Parallel| yes | no |

## GPU Implementations

For execution, please run the binary to get information regarding command line arguments.

Currently the following combinations are implemented:

| Functional Parallelism | Dense LU | Sparse LU | Batched LU |
|---|---|---|---|
|Sequential| yes | yes | yes | yes |
|Parallel| Partially | yes | yes |
|Side Notes| Nested parallelism not supported |  | |

## MATLAB Script

`checkSolution_60.m` can be used to validate a sample case with `-f 2 -m 2`.
