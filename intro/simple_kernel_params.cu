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

/* Kernel. The pointer to the buffer c is an address 
   in the GPU memory space */
__global__ void add( int a, int b, int *c ) {
    *c = a + b; /*write to GPU memory*/
}

int main( void ) {
    int c;
    int *dev_c;

    /* allocate space on GPU */
    HANDLE_ERROR( cudaMalloc( (void**)&dev_c, sizeof(int) ) );

    add<<<1,1>>>( 2, 7, dev_c );

    /* transfer memory from GPU to CPU
       dst, src, size, direction */
    HANDLE_ERROR( cudaMemcpy( &c, dev_c, sizeof(int),
                              cudaMemcpyDeviceToHost ) );
    printf( "2 + 7 = %d\n", c );

    /* free space on GPU */
    HANDLE_ERROR( cudaFree( dev_c ) );

    return 0;
}
