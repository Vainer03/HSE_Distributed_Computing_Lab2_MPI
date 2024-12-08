#include "task_1.h"

#define Lab_2_1
//#define Debug_Output

#ifdef Lab_2_1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

void initialize_vector(int n, int *vector) {
    if (!vector) {
        fprintf(stderr, "No vector submitted");
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
    else {
        for (int i = 0; i < n; i++) {
            vector[i] = i + 1;
        }
    }
}

void print_vector(int n, int *vector) {
    if (!vector) {
        fprintf(stderr, "No vector submitted");
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
    else {
        printf("Print Vector \n");
        for (int i = 0; i < n; i++) {
            printf("%d ", vector[i]);
        }
        printf("\n");
    }
}

void print_matrix(int n, int *matrix) {
    if (!matrix) {
        fprintf(stderr, "No matrix submitted");
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
    else {
        printf("Print Matrix \n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%d ", matrix[i * n + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void matrix_vector_multiplication_row_split(int comm_rank, int comm_size, int n) {
    int *matrix = (int *)malloc(n * n * sizeof(int));
    int *vector = (int *)malloc(n * sizeof(int));
    int *result = (int *)malloc(n * sizeof(int));

    if (comm_rank == 0) {
        initialize_vector(n * n, matrix);
        initialize_vector(n, vector);
#ifdef Debug_Output
        print_matrix(n, matrix);
        print_vector(n, vector);
#endif
    }

    int local_rows = n / comm_size + (comm_rank < n % comm_size ? 1 : 0);
    int *local_matrix = (int *)malloc(local_rows * n * sizeof(int));

    MPI_Bcast(vector, n, MPI_INT, 0, MPI_COMM_WORLD);

    int *send_counts = (int *)malloc(comm_size * sizeof(int));
    int *send_displacements = (int *)malloc(comm_size * sizeof(int));
    int *receive_counts = (int *)malloc(comm_size * sizeof(int));
    int *receive_displacements = (int *)malloc(comm_size * sizeof(int));
    for (int i = 0; i < comm_size; i++) {
        send_counts[i] = (n/comm_size + (i < n % comm_size ? 1 : 0)) * n;
        send_displacements[i] = (i * (n/comm_size) + (i < n % comm_size ? i : n % comm_size)) * n;
        receive_counts[i] = n/comm_size + (i < n % comm_size ? 1 : 0);
        receive_displacements[i] = i * (n/comm_size) + (i < n % comm_size ? i : n % comm_size);
    }

    MPI_Scatterv(matrix, send_counts, send_displacements, MPI_INT, local_matrix, local_rows * n, MPI_INT, 0, MPI_COMM_WORLD);

    int *local_result = (int *)malloc(local_rows * sizeof(int));
    for (int i = 0; i < local_rows; i++) {
        local_result[i] = 0;
        for (int j = 0; j < n; j++) {
            local_result[i] += local_matrix[i * n + j] * vector[j];
        }
    }

    MPI_Gatherv(local_result, local_rows, MPI_INT, result, receive_counts, receive_displacements, MPI_INT, 0, MPI_COMM_WORLD);

#ifdef Debug_Output
    if (comm_rank == 0) {
        print_vector(n, result);
    }
#endif
    free(matrix);
    free(vector);
    free(result);
    free(local_matrix);
    free(local_result);
    free(send_counts);
    free(send_displacements);
    free(receive_counts);
    free(receive_displacements);
}

void matrix_vector_multiplication_column_split(int comm_rank, int comm_size, int n) {
    int *matrix = (int *)malloc(n * n * sizeof(int));
    int *vector = (int *)malloc(n * sizeof(int));
    int *result = (int *)malloc(n * sizeof(int));

    if (comm_rank == 0) {
        initialize_vector(n * n, matrix);
        initialize_vector(n, vector);
#ifdef Debug_Output
        print_matrix(n, matrix);
        print_vector(n, vector);
#endif
    }

    int local_columns = n / comm_size + (comm_rank < n % comm_size ? 1 : 0);

    int *local_matrix = (int *)malloc(local_columns * n * sizeof(int));
    int *local_vector = (int *)malloc(local_columns * sizeof(int));

    int *send_counts = (int *)malloc(comm_size * sizeof(int));
    int *send_displacements = (int *)malloc(comm_size * sizeof(int));
    int *receive_counts = (int *)malloc(comm_size * sizeof(int));

    for (int i = 0; i < comm_size; i++) {
        send_counts[i] = n / comm_size + (i < n % comm_size ? 1 : 0);
        send_displacements[i] = i * n / comm_size + (i < n % comm_size ? i : n % comm_size);
    }

    MPI_Scatterv(vector, send_counts, send_displacements, MPI_INT, local_vector, local_columns, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Datatype MPI_column, MPI_resized_column;

    MPI_Type_vector(n, 1, n, MPI_INT, &MPI_column);
    MPI_Type_create_resized(MPI_column, 0, sizeof(int), &MPI_resized_column);
    MPI_Type_commit(&MPI_resized_column);

    MPI_Scatterv(matrix, send_counts, send_displacements, MPI_resized_column, local_matrix, local_columns * n, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Type_free(&MPI_resized_column);

    int *local_result = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        local_result[i] = 0;
    }

    for (int i = 0; i < comm_size; i++) {
        receive_counts[i] = n;
    }

    for (int i = 0; i < local_columns; i++) {
        for (int j = 0; j < n; j++) {
            local_result[j] += local_matrix[i * n + j] * local_vector[i];
        }
    }

    MPI_Reduce_scatter(local_result, result, receive_counts, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

#ifdef Debug_Output
    if (comm_rank == 0) {
        print_vector(n, result);
    }
#endif

    free(matrix);
    free(vector);
    free(result);
    free(local_matrix);
    free(local_vector);
    free(local_result);
    free(send_counts);
    free(send_displacements);
    free(receive_counts);
}

void matrix_vector_multiplication_block_split(int comm_rank, int comm_size, int n) {
    int grid_size = (int)sqrt((double)comm_size);
    if (grid_size * grid_size != comm_size) {
        if (comm_rank == 0) {
            fprintf(stderr, "Error: Number of processes must be a perfect square.\n");
        }
        return;
    }

    if (n % grid_size != 0) {
        if (comm_rank == 0) {
            fprintf(stderr, "Error: Matrix dimension must be divisible by sqrt(comm_size).\n");
        }
        return;
    }

    int block_size = n / grid_size;

    int *local_matrix = (int *)malloc(block_size * block_size * sizeof(int));
    int *local_vector = (int *)malloc(block_size * sizeof(int));
    int *local_result = (int *)malloc(block_size * sizeof(int));

    int *matrix = NULL;
    int *vector = NULL;
    int *result = NULL;

    if (comm_rank == 0) {
        matrix = (int *)malloc(n * n * sizeof(int));
        vector = (int *)malloc(n * sizeof(int));
        result = (int *)malloc(n * sizeof(int));

        for (int i = 0; i < n * n; i++) {
            matrix[i] = i + 1;
        }
        for (int i = 0; i < n; i++) {
            vector[i] = i + 1;
        }
    }

    {
        int local_size = block_size * block_size;
        int *temp_block = (int *)malloc(local_size * sizeof(int));

        if (comm_rank == 0) {
            for (int p = 0; p < comm_size; p++) {
                int start_row = (p / grid_size) * block_size;
                int start_col = (p % grid_size) * block_size;

                for (int row = 0; row < block_size; row++) {
                    for (int col = 0; col < block_size; col++) {
                        int global_row = start_row + row;
                        int global_col = start_col + col;
                        temp_block[row * block_size + col] = matrix[global_row * n + global_col];
                    }
                }

                if (p == 0) {
                    memcpy(local_matrix, temp_block, local_size * sizeof(int));
                } else {
                    MPI_Send(temp_block, local_size, MPI_INT, p, 0, MPI_COMM_WORLD);
                }
            }
        } else {
            MPI_Recv(local_matrix, block_size * block_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        free(temp_block);
    }

    {
        int *send_counts = NULL;
        int *displacements = NULL;

        if (comm_rank == 0) {
            send_counts = (int *)malloc(comm_size * sizeof(int));
            displacements = (int *)malloc(comm_size * sizeof(int));

            for (int p = 0; p < comm_size; p++) {
                send_counts[p] = block_size;
                displacements[p] = (p % grid_size) * block_size;
            }
        }

        MPI_Scatterv(vector, send_counts, displacements, MPI_INT,
                     local_vector, block_size, MPI_INT, 0, MPI_COMM_WORLD);

        if (comm_rank == 0) {
            free(send_counts);
            free(displacements);
        }
    }

    for (int i = 0; i < block_size; i++) {
        int sum = 0;
        for (int j = 0; j < block_size; j++) {
            sum += local_matrix[i * block_size + j] * local_vector[j];
        }
        local_result[i] = sum;
    }

    {
        int *partial_result = (int *)calloc(n, sizeof(int));
        int row_block_index = comm_rank / grid_size;

        for (int i = 0; i < block_size; i++) {
            int global_row = i + row_block_index * block_size;
            partial_result[global_row] = local_result[i];
        }

        MPI_Reduce(partial_result, (comm_rank == 0 ? result : NULL), n, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        free(partial_result);
    }

#ifdef Debug_Output
    if (comm_rank == 0) {
        print_vector(n, result);
    }
#endif

    free(local_matrix);
    free(local_vector);
    free(local_result);

    if (comm_rank == 0) {
        free(matrix);
        free(vector);
        free(result);
    }
}


int main(int argc, char** argv) {
    int comm_rank, comm_size;
    int multiplication_mode;
    double start, finish;
    long n_dim;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    if (argc != 3) {
        if (comm_rank == 0) {
            fprintf(stderr, "Use the following format: \n mpiexec -n n_threads %s n_matrix_dim multiplication_mode\n", argv[0]);
            MPI_Finalize();
            return EXIT_FAILURE;
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    n_dim = strtol(argv[1], NULL, 10);
    multiplication_mode = atoi(argv[2]);

    if (n_dim <= 0) {
        fprintf(stderr, "The matrix dimensions need to be > 0\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (n_dim < comm_size) {
        fprintf(stderr, "The matrix dimensions must be bigger than the number of processes\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    start = MPI_Wtime();
    switch (multiplication_mode) {
        case 1:
            matrix_vector_multiplication_row_split(comm_rank, comm_size, n_dim);
            break;
        case 2:
            matrix_vector_multiplication_column_split(comm_rank, comm_size, n_dim);
            break;
        case 3:
            matrix_vector_multiplication_block_split(comm_rank, comm_size, n_dim);
            break;
        default:
            fprintf(stderr, "Incorrect multiplication mode code\n");
            MPI_Finalize();
            return EXIT_FAILURE;
    }
    finish = MPI_Wtime();

    if (comm_rank == 0) {
        printf("\nTime taken: %lf seconds\n", finish - start);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}

#endif



