#ifndef TIMER_GPU_H
#define TIMER_GPU_H

#include "timer.h"

class GPUTimer: public TimerBase {
	private:
		cudaEvent_t startTime, stopTime;
	public:
		GPUTimer();
		~GPUTimer();
		void start(); 
		void stop();
		void print(); 
		float elapsed(); 
};

#endif
