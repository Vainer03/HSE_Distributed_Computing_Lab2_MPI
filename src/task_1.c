#include "task_1.h"

#define Lab_1_1

#ifdef Lab_1_1

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void initialize_vector(int n, int *vector) {
    if (!vector) {
        fprintf(stderr, "No vector submitted");
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

    int local_rows = n / comm_size + (comm_rank < n % comm_size ? 1 : 0);
    int *local_matrix = (int *)malloc(local_rows * n * sizeof(int));

    if (comm_rank == 0) {
        initialize_vector(n * n, matrix);
        initialize_vector(n, vector);
        print_matrix(n, matrix);
        print_vector(n, vector);
    }

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

    if (comm_rank == 0) {
        printf("\nResult:\n");
        for (int i = 0; i < n; i++) {
            printf("%d ", result[i]);
        }
    }
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

int main(int argc, char** argv) {
    int comm_rank, comm_size;
    double start, finish;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    if (argc != 2) {
        if (comm_rank == 0) {
            fprintf(stderr, "Use the following format: \n mpiexec -n n_threads %s n_matrix_dim\n", argv[0]);
            MPI_Finalize();
            return EXIT_FAILURE;
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    long n_dim = strtol(argv[1], NULL, 10);

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
    matrix_vector_multiplication_row_split(comm_rank, comm_size, n_dim);
    finish = MPI_Wtime();

    if (comm_rank == 0) {
        printf("\nTime taken: %lf seconds\n", finish - start);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}

#endif

