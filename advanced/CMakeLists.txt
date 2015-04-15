find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(Threads REQUIRED)

# Note set CUDA_BUILD_EMULATION to OFF to enable Device mode compilation.
#set(CUDA_SDK /usr/local/src/NVIDIA_CUDA_SDK)
#set(CUDA_INC ${CUDA_SDK}/common/inc/)

find_package(CUDA QUIET REQUIRED) #Any version OK

if (CUDA_FOUND)
   message("CUDA found")
else()
   message("CUDA not found")
endif()

cuda_include_directories(${CUDA_INC} ${CMAKE_CURRENT_SOURCE_DIR})

link_directories(${CUDA_SDK}/lib/)

set(HDRS book.h cpu_anim.h cpu_bitmap.h 
  gl_helper.h gpu_anim.h
  timer.h)

cuda_add_library(mycuda 
  timer.h book.h cudahelpers.h
  timer.cu book.cu cudahelpers.cu)


set(GPROG ripple shared_bitmap ray ray_noconst heat)

foreach(PROG ${GPROG})
  cuda_add_executable(${PROG} ${PROG}.cu ${HDRS})
  target_link_libraries(${PROG} 
    ${CMAKE_THREAD_LIBS_INIT}
    m mycuda 
    ${GLUT_LIBRARY} ${OPENGL_LIBRARY} 
    ${CMAKE_THREAD_LIBS_INIT})
endforeach()

#normal source files
set(SRC 
  main.cpp 
  mainwindow.cpp 
  mypanelopengl.cpp
)

set(SHADERS
  vshader.glsl
  fshader.glsl
)


