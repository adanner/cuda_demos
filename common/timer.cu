#include "timer_gpu.h"
#include <cstdio>

#define HANDLE_ERROR(a) a

using namespace std;

GPUTimer::GPUTimer() {
    HANDLE_ERROR( cudaEventCreate(&startTime) );
    HANDLE_ERROR( cudaEventCreate(&stopTime) );
}

GPUTimer::~GPUTimer(){
    HANDLE_ERROR( cudaEventDestroy(startTime) );
    HANDLE_ERROR( cudaEventDestroy(stopTime) );
}

void GPUTimer::start() {
    HANDLE_ERROR( cudaEventRecord(startTime,0) );
}

void GPUTimer::stop() {
    HANDLE_ERROR( cudaEventRecord(stopTime,0) );
    HANDLE_ERROR( cudaEventSynchronize(stopTime) );
}

void GPUTimer::print() {
    printf("GPU Time: %7.2f ms\n", this->elapsed());
}

float GPUTimer::elapsed() {
    float elapsed_ms;
    this->stop();
    HANDLE_ERROR( cudaEventElapsedTime(&elapsed_ms, startTime, stopTime) );
    return elapsed_ms/1000.; /*elapsed in timer.h is in seconds*/
}
