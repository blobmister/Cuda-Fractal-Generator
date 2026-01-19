cmake_minimum_required(VERSION 3.23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

## Replace with speciric architecture number
set(CMAKE_CUDA_ARCHITECTURES native)

project(Julia LANGUAGES CXX CUDA)

enable_language(CUDA)

add_executable(main main.cu)

add_library(head INTERFACE)

target_include_directories(head
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
)


target_sources(head
    INTERFACE
        FILE_SET HEADERS
        FILES
            complex.cuh
            fractal.cuh
            render.cuh
)

target_link_libraries(main
    PRIVATE
        head
)
