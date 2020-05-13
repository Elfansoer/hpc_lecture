#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  #pragma omp parallel for
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range); 
  
  //#pragma omp parallel for // (can be parallelized)
  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }
  
  #pragma omp parallel for
  for (int i=0; i<n; i++) {
    #pragma omp atomic update
    bucket[key[i]]++;
  }

  // for (int i=0, j=0; i<range; i++) {
  //   for (; bucket[i]>0; bucket[i]--) {
  //     key[j++] = i;
  //   }
  // }

  // rewrite code using different approach
  #pragma omp parallel for
  for( int i=0; i<range; i++ ) {
    int prev = 0

    #pragma omp parallel for reduction(+:prev)
    for( int j=0; j<i;j++ ) prev += bucket[j];
    
    #pragma omp parallel for
    for( int j=0; j<bucket[i]; j++ ) key[prev+j] = i;
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
