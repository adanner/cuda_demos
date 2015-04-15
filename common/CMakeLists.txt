#Build small common cuda library
cuda_add_library(common
  timer.h timer_gpu.h book.h timer.cu timer.cpp book.cu)

