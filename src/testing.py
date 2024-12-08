import subprocess
import matplotlib.pyplot as plt
import os
import numpy as np
import tqdm
import csv

def run_mpi_program(matrix_size, algorithm, num_processes, num_runs=5):
    times = []
    for _ in range(num_runs):
        try:
            result = subprocess.run(
                ["mpirun", "-np", str(num_processes), "./cmake-build-release/task_1", str(matrix_size), str(algorithm)],
                capture_output=True, text=True
            )
            # print(result.stdout)
            for line in result.stdout.splitlines():
                if "Time taken:" in line:
                    times.append(float(line.split(":")[1].strip().split()[0]))
        except Exception as e:
            print(f"Error running MPI program: {e}")
    return np.mean(times) if times else float('inf')

def calculate_speedup_and_efficiency(t1, tn, num_processes):
    speedup = t1 / tn if tn > 0 else 0
    efficiency = speedup / num_processes if num_processes > 0 else 0
    return speedup, efficiency

def save_results_to_csv(matrix_size, results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for alg, data in results.items():
        csv_file_path = os.path.join(save_dir, f'results_alg{alg}_size{matrix_size}.csv')
        with open(csv_file_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Processes", "Time (s)", "Speedup", "Efficiency"])
            for row in data:
                writer.writerow(row)
        print(f"Results for Algorithm {alg}, Matrix Size {matrix_size} saved to {csv_file_path}")

def plot_combined_results(matrix_size, results, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # Prepare data for plotting
    fig, (ax_time, ax_speedup, ax_efficiency) = plt.subplots(1, 3, figsize=(18, 6))

    for alg, data in results.items():
        processes, times, speedups, efficiencies = zip(*data)

        # Execution Time
        ax_time.plot(processes, times, marker='o', label=f'Algorithm {alg}')
        ax_time.set_title(f'Execution Time for Matrix Size {matrix_size}')
        ax_time.set_xlabel('Number of Processes')
        ax_time.set_ylabel('Time (seconds)')
        ax_time.grid(True)

        # Speedup
        ax_speedup.plot(processes, speedups, marker='o', label=f'Algorithm {alg}')
        ax_speedup.set_title(f'Speedup for Matrix Size {matrix_size}')
        ax_speedup.set_xlabel('Number of Processes')
        ax_speedup.set_ylabel('Speedup')
        ax_speedup.grid(True)

        # Efficiency
        ax_efficiency.plot(processes, efficiencies, marker='o', label=f'Algorithm {alg}')
        ax_efficiency.set_title(f'Efficiency for Matrix Size {matrix_size}')
        ax_efficiency.set_xlabel('Number of Processes')
        ax_efficiency.set_ylabel('Efficiency')
        ax_efficiency.grid(True)

    # Add legends
    ax_time.legend()
    ax_speedup.legend()
    ax_efficiency.legend()

    # Save the plot
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f'combined_plot_size_{matrix_size}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Combined plots for Matrix Size {matrix_size} saved to {plot_path}")


def main():
    matrix_sizes = [512, 1024, 2048, 4096]
    algorithms = {1: range(1, 7), 2: range(1, 7), 3: [1, 4]}
    num_runs = 5

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    for matrix_size in matrix_sizes:
        results = {alg: [] for alg in algorithms}
        for alg, process_counts in algorithms.items():
            t1 = run_mpi_program(matrix_size, alg, 1, num_runs)
            for num_processes in process_counts:
                tn = run_mpi_program(matrix_size, alg, num_processes, num_runs)
                speedup, efficiency = calculate_speedup_and_efficiency(t1, tn, num_processes)
                results[alg].append((num_processes, tn, speedup, efficiency))

        save_dir = os.path.join(results_dir, f'size_{matrix_size}')
        os.makedirs(save_dir, exist_ok=True)
        save_results_to_csv(matrix_size, results, save_dir)
        plot_combined_results(matrix_size, results, save_dir)

if __name__ == "__main__":
    main()
