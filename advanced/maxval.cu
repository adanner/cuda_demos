#include "book.h"
#include <climits>
#include <cstdlib>
#include <iostream>
#include "timer.h"

using namespace std;

#define imin(a,b) (a<b?a:b)

const int N = 33;
//const int threadsPerBlock = 256;
const int blocksPerGrid = 32;

float rand_in_range(float min, float max){
  return (max-min)*(1.0*(rand()%INT_MAX))/INT_MAX+min;
}

/* N is a global constant */
float max_cpu(float *a){
  float mval = a[0];
  for(int i=1; i<N; i++){
    if (a[i] > mval){
      mval = a[i];
    }
  }
  return mval;
}

int main( void ) {
  float   *a, *partial_result;
  float   *dev_a, *dev_result;

  GPUTimer gtime;

  /* Since blocks cannot share memory, if 
     multiple blocks are collaborating, they can copy
     partial results to a global array which is
     copied from GPU->CPU after the kernel is done.
     The CPU can then do some post processing on this 
     array of partial results. 

     You may want to change this value or have it depend on something other 
     than blocksPerGrid. 
   */
  int partial_size = blocksPerGrid;

  // allocate memory on the cpu side
  a = (float*)malloc( N*sizeof(float) );
  partial_result = (float*)malloc( partial_size*sizeof(float) );

  // allocate the memory on the GPU
  HANDLE_ERROR( cudaMalloc( (void**)&dev_a,
        N*sizeof(float) ) );
  HANDLE_ERROR( cudaMalloc( (void**)&dev_result,
        partial_size*sizeof(float) ) );

  // fill in the host memory with data
  for (int i=0; i<N; i++) {
    /* when making larger arrays, increase this range */
    a[i] = rand_in_range(0.0f,1000.0f);
    cout << i << " " << a[i] << endl;
  }

  //initialize partial results to be all 0
  for(int i=0; i<partial_size; i++){
    partial_result[i]=0.0f;
  }

  // copy the arrays 'a' and 'b' to the GPU
  HANDLE_ERROR( cudaMemcpy( dev_a, a, N*sizeof(float),
        cudaMemcpyHostToDevice ) );


  gtime.start();

  // call the kernel
  //max_gpu_single<<<1,1>>>( dev_a, dev_result );

  gtime.stop();
  printf("Time to run kernel on GPU: %7.2f ms\n", gtime.elapsed());

  // copy the array 'result' back from the GPU to the CPU
  HANDLE_ERROR( cudaMemcpy( partial_result, dev_result,
        partial_size*sizeof(float),
        cudaMemcpyDeviceToHost ) );

  CPUTimer t;
  t.start();

  // finish up on the CPU side
  float ans = partial_result[0];
  for (int i=0; i<partial_size; i++) {
    if(partial_result[i] > ans){
      ans = partial_result[i];
    }
  }

  cout << "Max (by GPU) " << ans << endl;
  t.stop();
  printf("Time to run CPU-GPU finishing: %7.2f ms\n", 1000*t.elapsed());

  t.start();
  cout << "Max (by CPU) " << max_cpu(a) << endl;
  t.stop();
  printf("Time to run on CPU: %7.2f ms\n", 1000*t.elapsed());

  // free memory on the gpu side
  HANDLE_ERROR( cudaFree( dev_a ) );
  HANDLE_ERROR( cudaFree( dev_result ) );

  // free memory on the cpu side
  free( a );
  free( partial_result );
}
