#pragma once

#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int create_matrix_memory(int ***matrix_ref, int rows, int cols);
int free_matrix_memory(int ***matrix_ref);
void init_matrix_data(int ***matrix_ref, int dim, int set_identity);
void generate_matrix(int ***matrix_ref, int dim, int create_identity);
void print_matrix(int **matrix, int dim);
void perform_block_multiplication(int **mat_x, int **mat_y, int block_rows, int block_cols, int ***mat_z);

int main(int argc, char *argv[]);
