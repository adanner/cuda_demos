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

/* Defining a kernel. The keyword __global__ 
   indicates this fucntion will be called from the host/CPU
   and run on the device/GPU. All kernels must have a 
   void return type */
__global__ void kernel( void ) {
   /* do nothing */
}

int main( void ) {
    /* 
     Calling the kernel,. The <<<gridDim,threadDim>>> syntax
     describes the number of blocks in a grid and number of threads
     in a block, respectively. The kernel will be executed in 
     parallel  gridDim * threadDim times on the GPU. 
    */ 
    kernel<<<1,1>>>();
    printf( "Hello, World!\n" );
    return 0;
}
