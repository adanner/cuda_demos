set(HDRS 
  ../common/book.h 
  ../common/cpu_anim.h 
  ../common/cpu_bitmap.h 
  ../common/gpu_anim.h)

SET(CPUTIMER
  ../common/timer.h
  ../common/timer.cpp
)

set(EASYPROGS hello_world 
  simple_kernel simple_kernel_params simple_device_call
  enum_gpu set_gpu add_loop_gpu add_loop_long
)

#Common Graphics Libraries
set(CGLLIBS ${GLUT_LIBRARY} ${OPENGL_LIBRARY} ${CMAKE_THREAD_LIBS_INIT} m)

#CS35 stuff. with CMake
add_executable(add_loop_cpu add_loop_cpu.cpp ${CPUTIMER})

#CS40 stuff. link against opengl
add_executable(julia_cpu julia_cpu.cpp 
  ${CPUTIMER} ../common/cpu_bitmap.h)
target_link_libraries(julia_cpu ${CGLLIBS})

#CUDA stuff, use cuda_add_executable 
foreach(PROG ${EASYPROGS})
 cuda_add_executable(${PROG} ${PROG}.cu ${HDRS} )
 target_link_libraries(${PROG} common ${CMAKE_THREAD_LIBS_INIT} )
endforeach()

cuda_add_executable(julia_gpu_timed julia_gpu_timed.cu ${HDRS})
target_link_libraries(julia_gpu_timed common ${CGLLIBS})

cuda_add_executable(julia_gpu julia_gpu.cu ${HDRS})
target_link_libraries(julia_gpu common ${CGLLIBS})

