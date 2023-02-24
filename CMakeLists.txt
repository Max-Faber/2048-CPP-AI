cmake_minimum_required(VERSION 3.24)
include(ExternalProject)
project(2048_CPP_AI)
set(CMAKE_CXX_STANDARD 17)

include_directories(src/include)
include_directories(src/include/AI)
include_directories(src/include/AI/Controllers)
include_directories(src/include/AI/Controllers/DQN)
include_directories(2048-CPP/src/include)
include_directories(2048-CPP/Libraries/include)
include_directories(2048-CPP/Libraries/include/FreeType)

get_filename_component(CUR_DIR "${CMAKE_CURRENT_LIST_DIR}/${MY_RELATIVE_PATH_VAR}" ABSOLUTE)
string(CONCAT CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,.")

add_executable(2048_CPP_AI
        src/2048-AI.cpp
        src/AI/Device.cpp
        src/AI/Controllers/DQN/DQN.cpp
        src/AI/Controllers/DQN/ExperienceReplay.cpp
        src/AI/Controllers/DQN/EpsilonAdaptiveGreedy.cpp
        src/AI/Controllers/DQN/Model.cpp
)
ExternalProject_Add(2048_CPP
        SOURCE_DIR ../2048-CPP
        CMAKE_ARGS -DLIB_SHARED=1
        BUILD_ALWAYS 1
        INSTALL_COMMAND ""
        BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/2048_CPP-prefix/src/2048_CPP-build/lib2048_CPP.dylib
)

ExternalProject_Add(libtorch
        SOURCE_DIR ../pytorch
        CMAKE_ARGS -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Release -DUSE_CUDA=0 -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_BINARY_DIR}/pytorch-install ${CMAKE_BINARY_DIR}/libtorch-prefix
        BUILD_ALWAYS 1
        BUILD_COMMAND cmake --build . --target install
        BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/libtorch-prefix/src/libtorch-build/lib/libtorch_cpu.dylib ${CMAKE_BINARY_DIR}/libtorch-prefix/src/libtorch-build/lib/libc10.dylib
)

if (APPLE)
    include_directories(${CMAKE_BINARY_DIR}/pytorch-install/include)
    include_directories(${CMAKE_BINARY_DIR}/pytorch-install/include/torch/csrc/api/include)

    set(TORCH ${CMAKE_BINARY_DIR}/pytorch-install/lib/libtorch_cpu.dylib)
    set(C10 ${CMAKE_BINARY_DIR}/pytorch-install/lib/libc10.dylib)
endif ()

ExternalProject_Get_Property(2048_CPP BINARY_DIR)

add_custom_command(
        TARGET ${PROJECT_NAME} PRE_LINK
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${BINARY_DIR}/lib2048_CPP.dylib
        ${CMAKE_CURRENT_BINARY_DIR}/lib2048_CPP.dylib
)

add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_CURRENT_BINARY_DIR}/fonts
)

add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${BINARY_DIR}/fonts/arial.ttf
        ${CMAKE_CURRENT_BINARY_DIR}/fonts/arial.ttf
)

add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${BINARY_DIR}/fonts/SmallMemory.ttf
        ${CMAKE_CURRENT_BINARY_DIR}/fonts/SmallMemory.ttf
)
# TODO: Replace the 3 add_custom_command calls with the one below if Homebrew rolls out cmake 3.26 (adds copy_directory_if_different functionality)
#add_custom_command(
#        TARGET ${PROJECT_NAME} PRE_BUILD
#        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
#        ${BINARY_DIR}/fonts/
#        ${CMAKE_CURRENT_BINARY_DIR}/fonts
#)

target_link_libraries(2048_CPP_AI
        #        ${BINARY_DIR}/lib2048_CPP.dylib
        ${CMAKE_BINARY_DIR}/2048_CPP-prefix/src/2048_CPP-build/lib2048_CPP.dylib
        ${TORCH}
        ${C10}
)

