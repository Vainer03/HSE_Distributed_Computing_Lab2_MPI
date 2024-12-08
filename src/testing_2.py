import subprocess
import matplotlib.pyplot as plt
import os
import numpy as np
import csv
import tqdm
from math import sqrt, floor

def is_perfect_square(num):
    root = int(sqrt(num))
    return root * root == num

def run_mpi_program(matrix_size, num_processes, num_runs=5):
    times = []
    for _ in range(num_runs):
        try:
            result = subprocess.run(
                ["mpirun", "-np", str(num_processes), "./cmake-build-release/task_2", str(matrix_size)],
                capture_output=True, text=True
            )
            for line in result.stdout.splitlines():
                if "Time:" in line:
                    times.append(float(line.split(":")[1].strip()))
        except Exception as e:
            print(f"Error running MPI program: {e}")
    return np.mean(times) if times else float('inf')

def calculate_speedup_and_efficiency(t1, tn, num_processes):
    speedup = t1 / tn if tn > 0 else 0
    efficiency = speedup / num_processes if num_processes > 0 else 0
    return speedup, efficiency

def save_results_to_csv(matrix_size, results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    csv_file_path = os.path.join(save_dir, f'results_size_{matrix_size}.csv')
    with open(csv_file_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Processes", "Time (s)", "Speedup", "Efficiency"])
        for row in results:
            writer.writerow(row)
    print(f"Results for Matrix Size {matrix_size} saved to {csv_file_path}")

def plot_combined_results(matrix_size, results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    processes, times, speedups, efficiencies = zip(*results)

    fig, (ax_time, ax_speedup, ax_efficiency) = plt.subplots(1, 3, figsize=(18, 6))

    # Execution Time
    ax_time.plot(processes, times, marker='o', label='Execution Time', color='blue')
    ax_time.set_title(f'Execution Time for Matrix Size {matrix_size}')
    ax_time.set_xlabel('Number of Processes')
    ax_time.set_ylabel('Time (seconds)')
    ax_time.grid(True)

    # Speedup
    ax_speedup.plot(processes, speedups, marker='o', label='Speedup', color='green')
    ax_speedup.set_title(f'Speedup for Matrix Size {matrix_size}')
    ax_speedup.set_xlabel('Number of Processes')
    ax_speedup.set_ylabel('Speedup')
    ax_speedup.grid(True)

    # Efficiency
    ax_efficiency.plot(processes, efficiencies, marker='o', label='Efficiency', color='red')
    ax_efficiency.set_title(f'Efficiency for Matrix Size {matrix_size}')
    ax_efficiency.set_xlabel('Number of Processes')
    ax_efficiency.set_ylabel('Efficiency')
    ax_efficiency.grid(True)

    # Save the combined plot
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f'combined_plot_size_{matrix_size}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Combined plots for Matrix Size {matrix_size} saved to {plot_path}")


def main():
    matrix_sizes = [128, 256, 512, 1024]
    max_processes = 4  # Adjust if you have more processes available
    num_runs = 5

    results_dir = "results_task_2"
    os.makedirs(results_dir, exist_ok=True)

    for matrix_size in matrix_sizes:
        results = []
        t1 = run_mpi_program(matrix_size, 1, num_runs)  # Baseline time with 1 process
        for num_processes in range(1, max_processes + 1):
            if is_perfect_square(num_processes) and matrix_size % sqrt(num_processes) == 0:
                tn = run_mpi_program(matrix_size, num_processes, num_runs)
                speedup, efficiency = calculate_speedup_and_efficiency(t1, tn, num_processes)
                results.append((num_processes, tn, speedup, efficiency))

        save_dir = os.path.join(results_dir, f'size_{matrix_size}')
        save_results_to_csv(matrix_size, results, save_dir)
        plot_combined_results(matrix_size, results, save_dir)

if __name__ == "__main__":
    main()
