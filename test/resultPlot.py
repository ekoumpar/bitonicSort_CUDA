import os
import matplotlib.pyplot as plt

def parse_results(file_path, version):
    results = {}
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                if line.startswith("Running") and version in line:
                    q = int(line.split('=')[1].split('...')[0])
                    time_line = lines[i + 2]  # Adjusted to skip the "Sorted array" line
                    time = float(time_line.split(': ')[1].split(' ')[0])
                    results[q] = time
    except Exception as e:
        print(f"Error parsing results for {version}: {e}")
    return results

def parse_qsort_results(file_path):
    results = {}
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith("q="):
                    q = int(line.split('=')[1].split('\t')[0])
                    time = float(line.split(': ')[1].split(' ')[0])
                    results[q] = time
    except Exception as e:
        print(f"Error parsing qSort results: {e}")
    return results

def plot_results():
    versions = ['V0', 'V1', 'V2']
    results_file = os.path.join(os.path.dirname(__file__), 'aristotelis_results.txt')
    qsort_results_file = os.path.join(os.path.dirname(__file__), '..', 'qSort', 'qSort_results.txt')

    all_results = {}
    for version in versions:
        all_results[version] = parse_results(results_file, version)
    
    all_results['qSort'] = parse_qsort_results(qsort_results_file)

    # Plot for V0, V1, and V2
    plt.figure(figsize=(10, 6))
    for version in versions:
        results = all_results[version]
        qs = sorted(results.keys())
        times = [results[q] for q in qs]
        plt.plot(qs, times, label=version, marker='o')

    plt.xlabel('Value of q (Sorting 2^q number of elements)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time vs Value of q for Versions V0, V1, V2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    output_file_v = os.path.join(os.path.dirname(__file__), '..', 'documentation', 'assets', 'results_comparison_versions.png')
    plt.savefig(output_file_v)
    plt.show()

    # Plot for qSort
    plt.figure(figsize=(10, 6))
    results = all_results['qSort']
    qs = sorted(results.keys())
    times = [results[q] for q in qs]
    plt.plot(qs, times, label='qSort', marker='o')

    plt.xlabel('Value of q (Sorting 2^q number of elements)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time vs Value of q for qSort')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    output_file_qsort = os.path.join(os.path.dirname(__file__), '..', 'documentation', 'assets', 'results_comparison_qsort.png')
    plt.savefig(output_file_qsort)
    plt.show()

if __name__ == "__main__":
    try:
        plot_results()
    except Exception as e:
        print(f"Error during plot_results execution: {e}")