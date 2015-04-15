/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */


#include "common/book.h"
#include "common/timer_gpu.h"

#define N   (32 * 1024 * 1024)

/* Block level kernel. Logically, a grid is composed of many
   blocks.  Physically, each block is assigned to a single
   Multiprocessor (MP) on CUDA Hardware. The Jetson TK1
   has only a single MP. */
__global__ void add_block( int *a, int *b, int *c ) {
    int tid = blockIdx.x;
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        tid += gridDim.x;
    }
}

/* Thread level kernel. Logically, a block is composed of many
   threads.  Physically, threads are assigned to cores
   within a Multiprocessor (MP) on CUDA Hardware. The Jetson TK1
   has 192 cores/MP.  Groups of 32 threads are called a warp. 
   All threads within a warp must execute the same instruction or 
   remain idle.
 */
__global__ void add_thread( int *a, int *b, int *c ) {
    int tid = threadIdx.x;
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x;
    }
}

/* a single threaded CPU kernel */
void add_cpu( int *a, int *b, int *c ) {
  int tid = 0;
  while (tid < N){
    c[tid] = a[tid] + b[tid];
    tid += 1;
  }
}

int main( void ) {
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;
    GPUTimer gt;
    CPUTimer ct;

    // allocate the memory on the CPU
    a = (int*)malloc( N * sizeof(int) );
    b = (int*)malloc( N * sizeof(int) );
    c = (int*)malloc( N * sizeof(int) );

    // allocate the memory on the GPU
    HANDLE_ERROR( cudaMalloc( (void**)&dev_a, N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_b, N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_c, N * sizeof(int) ) );

    // fill the arrays 'a' and 'b' on the CPU
    for (int i=0; i<N; i++) {
        a[i] = i;
        b[i] = 2 * i;
    }

    // copy the arrays 'a' and 'b' to the GPU
    HANDLE_ERROR( cudaMemcpy( dev_a, a, N * sizeof(int),
                              cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_b, b, N * sizeof(int),
                              cudaMemcpyHostToDevice ) );
    
    /*time the GPU kernel using GPU timer*/
    gt.start();
    add_block<<<128,1>>>( dev_a, dev_b, dev_c );
    gt.stop();
    printf("Time to run block kernel: %3.1f ms\n", 1000*gt.elapsed());

    /* *
    gt.start();
    add_thread<<<1,128>>>( dev_a, dev_b, dev_c );
    gt.stop();
    printf("Time to run thread kernel: %3.1f ms\n", 1000*gt.elapsed());
    // */

    // copy the array 'c' back from the GPU to the CPU
    HANDLE_ERROR( cudaMemcpy( c, dev_c, N * sizeof(int),
                              cudaMemcpyDeviceToHost ) );

    // verify that the GPU did the work we requested
    bool success = true;
    for (int i=0; i<N; i++) {
        if ((a[i] + b[i]) != c[i]) {
            printf( "Error:  %d + %d != %d\n", a[i], b[i], c[i] );
            success = false;
        }
    }
    if (success) { 
      printf( "We did it!\n" );
    }
    // free the memory we allocated on the GPU
    HANDLE_ERROR( cudaFree( dev_a ) );
    HANDLE_ERROR( cudaFree( dev_b ) );
    HANDLE_ERROR( cudaFree( dev_c ) );

    /*time the CPU version using CPU timer*/
    ct.start();
    add_cpu(a,b,c);
    ct.stop();
    printf("Time to run on CPU: %3.1f ms\n", 1000*ct.elapsed());



    // free the memory we allocated on the CPU
    free( a );
    free( b );
    free( c );

    return 0;
}

