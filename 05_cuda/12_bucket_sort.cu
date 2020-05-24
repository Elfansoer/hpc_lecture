#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void bucketsort(int *a, int n, int range) {
  // init identifier
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i>=n) return;

  // init bucket
  extern __shared__ int bucket[];
  __syncthreads();
  if (threadIdx.x<range)
    bucket[threadIdx.x] = 0;

  // add to bucket
  __syncthreads();
  atomicAdd( &bucket[a[i]], 1 );
  __syncthreads();

  // prefix sum for each bucket
  int pos = 0;
  int prefix = 0;
  while(prefix + bucket[pos] <= i) {
    prefix += bucket[pos];
    pos++;
  }
  
  // spread bucket
  a[i] = pos;
}

int main() {
  int n = 50;
  int range = 5;

  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  // copy to unified memory, leaving original vector intact
  int *a;
  cudaMallocManaged(&a, n*sizeof(int));
  for(int i=0;i<n;i++) a[i] = key[i];
  
  // call gpu
  bucketsort<<<1,n,range>>>(a, n, range);
  cudaDeviceSynchronize();

  // copy back to vector
  for (int i=0; i<n; i++) {
    key[i] = a[i];
    printf("%d ",key[i]);
  }
  printf("\n");
}
