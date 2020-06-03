#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#define DEBUG

#include <gemm/dispatch.h>
#include <gemm/epilogue_function.h>
//#include "util/matrix.h"
#include "util/timer.h"

using namespace cutlass;

// for Windows testing
float drand48() {
  return rand() / (RAND_MAX + 1.0);
}

int main(int argc, const char **argv) {
  // consts
  int m = 10240;
  int k = 4096;
  int n = 4096;
  float alpha = 1.0;
  float beta = 0.0;
  int g_timing_iterations = 10;

  // static const matrix_transform_t::kind_t TransformA = matrix_transform_t::NonTranspose;
  // static const matrix_transform_t::kind_t TransformB = matrix_transform_t::NonTranspose;
  static const int TransformA = 0;
  static const int TransformB = 0;

  // typedef float value_t;
  // typedef float accum_t;
  cudaStream_t stream = 0;

  // definitions
  float *A, *B, *C, *C2;
  cudaMallocManaged(&A, m*k * sizeof(float) );
  cudaMallocManaged(&B, k*n * sizeof(float) );
  cudaMallocManaged(&C, m*n * sizeof(float) );
  cudaMallocManaged(&C2, m*n * sizeof(float) );
  // matrix<float> A(m, k);
  // matrix<float> B(k, n);
  // matrix<float> C(m, n);
  // matrix<float> C2(m, n);

  // fill out
  for( int jndex=0; jndex<k; jndex++ ) {
    for(int index=0; index<m; index++) {
      A[jndex*m + index] = drand48();
    }
  }
  for( int jndex=0; jndex<n; jndex++ ) {
    for(int index=0; index<k; index++) {
      B[jndex*k + index] = drand48();
    }
  }
  for( int jndex=0; jndex<n; jndex++ ) {
    for(int index=0; index<m; index++) {
      C[jndex*m + index] = 0;
      C2[jndex*m + index] = 0;
    }
  }
  cudaDeviceSynchronize();  
  // A.random();
  // B.random();
  // C.fill_ramp(0,0);
  // C2.fill_ramp(0,0);
  // A.sync_device();
  // B.sync_device();
  // C.sync_device();
  // C2.sync_device();

  // CUBLAS
  cublasHandle_t g_cublas_handle;
  cublasCreate(&g_cublas_handle);
  gpu_timer timer;
  for (int i = 0; i < g_timing_iterations+2; i++) {
    if (i == 2) timer.start();
    // CUDA_PERROR(cublasSgemm(
    //                         g_cublas_handle,
    //                         (cublasOperation_t) TransformA,
    //                         (cublasOperation_t) TransformB,
    //                         m,
    //                         n,
    //                         k,
    //                         &alpha,
    //                         A.d_data(),
    //                         m,
    //                         B.d_data(),
    //                         k,
    //                         &beta,
    //                         C.d_data(),
    //                         m));
    CUDA_PERROR(cublasSgemm(
                            g_cublas_handle,
                            (cublasOperation_t) TransformA,
                            (cublasOperation_t) TransformB,
                            m,
                            n,
                            k,
                            &alpha,
                            A,
                            m,
                            B,
                            k,
                            &beta,
                            C,
                            m));
  }
  timer.stop();

  // calculate CUBLAS time
  int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
  double tcublas = timer.elapsed_millis() / g_timing_iterations;
  double cublas_flops = double(num_flops) / tcublas / 1.0e6;

  // CUTLASS
  typedef gemm::blas_scaled_epilogue<float, float, float> epilogue_op_t;
  epilogue_op_t epilogue(alpha, beta);
  for (int i = 0; i < g_timing_iterations+2; i++) {
    if (i == 2) timer.start();
    // gemm::dispatch<epilogue_op_t>(
    //     m,
    //     n,
    //     k,
    //     alpha,
    //     beta,
    //     A.d_data(),
    //     B.d_data(),
    //     C2.d_data(),
    //     stream,
    //     false);
    gemm::dispatch<epilogue_op_t>(
        m,
        n,
        k,
        alpha,
        beta,
        A,
        B,
        C2,
        stream,
        false);
  }
  timer.stop();

  // calculate CUTLASS time
  double tcutlass = timer.elapsed_millis() / g_timing_iterations;
  double cutlass_flops = double(num_flops) / tcutlass / 1.0e6;

  // error performance summary. No need to optimize below this line
  printf("CUBLAS: %.2f Gflops, CUTLASS: %.2f Gflops\n", cublas_flops, cutlass_flops);
  // C.sync_host();
  // C2.sync_host();
  cudaDeviceSynchronize();  

  double err = 0;
  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++) {
      // err += fabs(C.get(i,j) - C2.get(i,j));
      err += fabs(C[i*m + j] - C2[i*m + j]);
    }
  }
  printf("error: %lf\n", err/n/m);
  cublasDestroy(g_cublas_handle);
}
