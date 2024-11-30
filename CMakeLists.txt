cmake_minimum_required(VERSION 3.13)
project(HSE_DC_Lab2_CLion C)

set(CMAKE_C_STANDARD 11)

find_package(MPI REQUIRED)

add_executable(HSE_DC_Lab2_CLion
        src/main.c)
target_link_libraries(HSE_DC_Lab2_CLion PUBLIC MPI::MPI_C)

add_executable(task_1
        src/task_1.c
        src/task_1.h)
target_link_libraries(task_1 PUBLIC MPI::MPI_C)

add_executable(task_2
        src/task_2.c
        src/task_2.h)
target_link_libraries(task_2 PUBLIC MPI::MPI_C)

add_executable(task_3
        src/task_3.c
        src/task_3.h)
target_link_libraries(task_3 PUBLIC MPI::MPI_C)
