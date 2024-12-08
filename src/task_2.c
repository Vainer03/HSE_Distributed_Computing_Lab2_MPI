#include "task_2.h"

#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int create_matrix_memory(int ***matrix_ref, int rows, int cols) {
    int *data = (int *)malloc(sizeof(int) * rows * cols);
    if (!data) {
        return -1;
    }
    *matrix_ref = (int **)malloc(rows * sizeof(int *));
    if (!*matrix_ref) {
        free(data);
        return -1;
    }
    for (int i = 0; i < rows; i++) {
        (*matrix_ref)[i] = &data[i * cols];
    }
    return 0;
}

int free_matrix_memory(int ***matrix_ref) {
    free(&((*matrix_ref)[0][0]));
    free(*matrix_ref);
    return 0;
}

void init_matrix_data(int ***matrix_ref, int dim, int set_identity) {
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            if (set_identity && i == j) {
                (*matrix_ref)[i][j] = 1;
            } else if (set_identity) {
                (*matrix_ref)[i][j] = 0;
            } else {
                (*matrix_ref)[i][j] = rand() % 10;
            }
        }
    }
}

void generate_matrix(int ***matrix_ref, int dim, int create_identity) {
    if (create_matrix_memory(matrix_ref, dim, dim) != 0) {
        printf("Matrix allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    init_matrix_data(matrix_ref, dim, create_identity);
}

void print_matrix(int **matrix, int dim) {
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

void perform_block_multiplication(int **mat_x, int **mat_y, int block_rows, int block_cols, int ***mat_z) {
    for (int i = 0; i < block_rows; i++) {
        for (int j = 0; j < block_cols; j++) {
            int val = 0;
            for (int k = 0; k < block_rows; k++) {
                val += mat_x[i][k] * mat_y[k][j];
            }
            (*mat_z)[i][j] = val;
        }
    }
}


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int comm_size, comm_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    int matrix_dim = atoi(argv[1]);
    if (!matrix_dim) {
        if (comm_rank == 0) {
            printf("Usage: mpirun -n num_processes ./cmake-build-release/task_2 matrix_size\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int is_b_identity = 1;
    int global_rows = matrix_dim;
    int global_cols = matrix_dim;

    int **mat_a = NULL, **mat_b = NULL, **mat_c = NULL;
    int grid_dim, block_dim;

    if (comm_rank == 0) {
        double root_comm_size = sqrt(comm_size);
        if ((root_comm_size - floor(root_comm_size)) != 0) {
            printf("Number of processes must be a perfect square\n");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }

        int procs_per_dim = (int)root_comm_size;
        if (global_cols % procs_per_dim != 0 || global_rows % procs_per_dim != 0) {
            printf("Matrix dimension not divisible by %d\n", procs_per_dim);
            MPI_Abort(MPI_COMM_WORLD, 3);
        }

        grid_dim = procs_per_dim;
        block_dim = global_cols / procs_per_dim;

        srand(time(NULL));
        generate_matrix(&mat_a, global_rows, 0);
        generate_matrix(&mat_b, global_rows, is_b_identity);

        if (create_matrix_memory(&mat_c, global_rows, global_cols) != 0) {
            printf("Matrix C allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 4);
        }
    }

    double start_time = MPI_Wtime();

    int broadcast_data[4];
    if (comm_rank == 0) {
        broadcast_data[0] = grid_dim;
        broadcast_data[1] = block_dim;
        broadcast_data[2] = global_rows;
        broadcast_data[3] = global_cols;
    }

    MPI_Bcast(&broadcast_data, 4, MPI_INT, 0, MPI_COMM_WORLD);
    grid_dim = broadcast_data[0];
    block_dim = broadcast_data[1];
    global_rows = broadcast_data[2];
    global_cols = broadcast_data[3];

    int dims[2], periods[2];
    dims[0] = grid_dim;
    dims[1] = grid_dim;
    periods[0] = 1;
    periods[1] = 1;
    int reorder = 1;

    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);

    int **local_a = NULL, **local_b = NULL, **local_c = NULL;
    create_matrix_memory(&local_a, block_dim, block_dim);
    create_matrix_memory(&local_b, block_dim, block_dim);

    int global_size[2] = { global_rows, global_cols };
    int local_size[2] = { block_dim, block_dim };
    int starts[2] = { 0, 0 };

    MPI_Datatype block_type, resized_block_type;
    MPI_Type_create_subarray(2, global_size, local_size, starts, MPI_ORDER_C, MPI_INT, &block_type);
    MPI_Type_create_resized(block_type, 0, block_dim * sizeof(int), &resized_block_type);
    MPI_Type_commit(&resized_block_type);

    int *a_data_ptr = NULL;
    int *b_data_ptr = NULL;
    int *c_data_ptr = NULL;

    if (comm_rank == 0) {
        a_data_ptr = &(mat_a[0][0]);
        b_data_ptr = &(mat_b[0][0]);
        c_data_ptr = &(mat_c[0][0]);
    }

    int *send_counts = NULL;
    int *displacements = NULL;
    if (comm_rank == 0) {
        send_counts = (int *)malloc(sizeof(int) * comm_size);
        displacements = (int *)malloc(sizeof(int) * comm_size);
        for (int i = 0; i < comm_size; i++)
            send_counts[i] = 1;

        int disp_count = 0;
        for (int i = 0; i < grid_dim; i++) {
            for (int j = 0; j < grid_dim; j++) {
                displacements[i * grid_dim + j] = disp_count;
                disp_count += 1;
            }
            disp_count += (block_dim - 1) * grid_dim;
        }
    }

    MPI_Scatterv(a_data_ptr, send_counts, displacements, resized_block_type,
                 &(local_a[0][0]), (global_rows * global_cols) / comm_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(b_data_ptr, send_counts, displacements, resized_block_type,
                 &(local_b[0][0]), (global_rows * global_cols) / comm_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (create_matrix_memory(&local_c, block_dim, block_dim) != 0) {
        printf("Allocation for local C failed at rank %d\n", comm_rank);
        MPI_Abort(MPI_COMM_WORLD, 7);
    }

    int coords[2];
    MPI_Cart_coords(cart_comm, comm_rank, 2, coords);

    int left_nb, right_nb, top_nb, bottom_nb;
    MPI_Cart_shift(cart_comm, 1, coords[0], &left_nb, &right_nb);
    MPI_Sendrecv_replace(&(local_a[0][0]), block_dim * block_dim, MPI_INT,
                         left_nb, 1, right_nb, 1, cart_comm, MPI_STATUS_IGNORE);

    MPI_Cart_shift(cart_comm, 0, coords[1], &top_nb, &bottom_nb);
    MPI_Sendrecv_replace(&(local_b[0][0]), block_dim * block_dim, MPI_INT,
                         top_nb, 1, bottom_nb, 1, cart_comm, MPI_STATUS_IGNORE);

    for (int i = 0; i < block_dim; i++) {
        for (int j = 0; j < block_dim; j++) {
            local_c[i][j] = 0;
        }
    }

    int **temp_result = NULL;
    if (create_matrix_memory(&temp_result, block_dim, block_dim) != 0) {
        printf("Allocation for temp result failed at rank %d\n", comm_rank);
        MPI_Abort(MPI_COMM_WORLD, 8);
    }

    for (int shift_count = 0; shift_count < grid_dim; shift_count++) {
        perform_block_multiplication(local_a, local_b, block_dim, block_dim, &temp_result);
        for (int i = 0; i < block_dim; i++) {
            for (int j = 0; j < block_dim; j++) {
                local_c[i][j] += temp_result[i][j];
            }
        }
        MPI_Cart_shift(cart_comm, 1, 1, &left_nb, &right_nb);
        MPI_Cart_shift(cart_comm, 0, 1, &top_nb, &bottom_nb);

        MPI_Sendrecv_replace(&(local_a[0][0]), block_dim * block_dim, MPI_INT,
                             left_nb, 1, right_nb, 1, cart_comm, MPI_STATUS_IGNORE);

        MPI_Sendrecv_replace(&(local_b[0][0]), block_dim * block_dim, MPI_INT,
                             top_nb, 1, bottom_nb, 1, cart_comm, MPI_STATUS_IGNORE);
    }

    MPI_Gatherv(&(local_c[0][0]), (global_rows * global_cols) / comm_size, MPI_INT,
                c_data_ptr, send_counts, displacements, resized_block_type, 0, MPI_COMM_WORLD);

    if (comm_rank == 0) {
        printf("Time: %.6f\n", MPI_Wtime() - start_time);
    }

    free_matrix_memory(&local_c);
    free_matrix_memory(&temp_result);

    MPI_Finalize();
    return 0;
}
