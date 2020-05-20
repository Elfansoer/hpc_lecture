#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

// for Windows testing
// float drand48() {
//   return rand() / (RAND_MAX + 1.0);
// }

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];

  float a[N]; // for mask
  float b[N]; // temp store

  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;

    // for mask
    a[i] = i;
  }

  // init mask & mass
  __m256 zmask = _mm256_set1_ps( 1 );
  __m256 omask = _mm256_setzero_ps();
  __m256 imask = _mm256_load_ps( a );
  __m256 mmvec = _mm256_load_ps( m );

  for(int i=0; i<N; i++) {
    // create 2 masks; equal and not equal
    __m256 jmask = _mm256_set1_ps( i );
    __m256 nmask = _mm256_cmp_ps( imask, jmask, _CMP_NEQ_OQ );
    nmask = _mm256_blendv_ps(omask, zmask, nmask);
    __m256 emask = _mm256_cmp_ps( imask, jmask, _CMP_EQ_OQ );
    emask = _mm256_blendv_ps(omask, zmask, emask);

    // init vectors
    __m256 xivec = _mm256_set1_ps( x[i] );
    __m256 yivec = _mm256_set1_ps( y[i] );
    __m256 xjvec = _mm256_load_ps( x );
    __m256 yjvec = _mm256_load_ps( y );

    // sub, square, add
    __m256 rxvec = _mm256_sub_ps( xivec, xjvec );
    __m256 x2vec = _mm256_mul_ps( rxvec, rxvec );
    __m256 ryvec = _mm256_sub_ps( yivec, yjvec );
    __m256 y2vec = _mm256_mul_ps( ryvec, ryvec );
    __m256 xyvec = _mm256_add_ps( x2vec, y2vec );

    // add the i==j element with 1 to avoid div by 0 using mask
    __m256 rrvec = _mm256_add_ps( xyvec, emask );

    // calculate 1/r, pow3, and mask
    rrvec = _mm256_rsqrt_ps( rrvec );
    __m256 temp1 = rrvec;
    rrvec = _mm256_mul_ps( rrvec, temp1 );
    rrvec = _mm256_mul_ps( rrvec, temp1 );
    rrvec = _mm256_mul_ps( rrvec, nmask );

    // calculate sum of force
    __m256 xysum = _mm256_mul_ps( mmvec, rrvec );
    __m256 xisum = _mm256_mul_ps( rxvec, xysum );
    __m256 yisum = _mm256_mul_ps( ryvec, xysum );

    // reduce, substract, store fx
    __m256 xrsum = _mm256_permute2f128_ps(xisum,xisum,1);
    xrsum = _mm256_add_ps(xrsum,xisum);
    xrsum = _mm256_hadd_ps(xrsum,xrsum);
    xrsum = _mm256_hadd_ps(xrsum,xrsum);
    _mm256_store_ps(b, xrsum);
    fx[i] -= b[0];

    // reduce, substract, store fy
    __m256 yrsum = _mm256_permute2f128_ps(yisum,yisum,1);
    yrsum = _mm256_add_ps(yrsum,yisum);
    yrsum = _mm256_hadd_ps(yrsum,yrsum);
    yrsum = _mm256_hadd_ps(yrsum,yrsum);
    _mm256_store_ps(b, yrsum);
    fy[i] -= b[0];

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}