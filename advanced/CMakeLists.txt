set(HDRS 
  ../common/book.h 
  ../common/cpu_anim.h 
  ../common/cpu_bitmap.h 
  ../common/gl_helper.h 
  ../common/gpu_anim.h
  ../common/timer.h
  ../common/timer_gpu.h
)

set(EASYPROG dot maxval)
foreach(PROG ${EASYPROG})
  cuda_add_executable(${PROG} ${PROG}.cu ${HDRS})
  target_link_libraries(${PROG} m common
    ${CMAKE_THREAD_LIBS_INIT})
endforeach()

set(GPROG ripple shared_bitmap ray ray_noconst heat)

foreach(PROG ${GPROG})
  cuda_add_executable(${PROG} ${PROG}.cu ${HDRS})
  target_link_libraries(${PROG} 
    m common 
    ${GLUT_LIBRARY} ${OPENGL_LIBRARY} 
    ${CMAKE_THREAD_LIBS_INIT})
endforeach()

