#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void initialize_vector(int n, int *vector); // can be used on a matrix
void print_vector(int n, int *vector);
void print_matrix(int n, int *matrix);

void matrix_vector_multiplication_row_split(int comm_rank, int comm_size, int n);
void matrix_vector_multiplication_column_split(int comm_rank, int comm_size, int n);
void matrix_vector_multiplication_block_split(int comm_rank, int comm_size, int n);

int main(int argc, char** argv);