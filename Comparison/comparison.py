import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import glob
import re
import sys
import json

# Set matplotlib backend
matplotlib.use('Agg')
try:
    import matplotlib.pyplot as plt
    plt.switch_backend('TkAgg')
except:
    pass

# Get project root directory path and add to sys.path
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Project root directory: {root_path}")
sys.path.append(root_path)

# Import necessary configurations and modules
from config import NUM_MEA, NUM_ROUNDS, NUM_QUBITS, SHOTS, SHOTS_EXP

# Import CMA optimization model generation modules
from Simulation.Creation_and_Sampling.sampling import run_sampling
from Simulation.Analysis_and_Plotting.analysis import compute_correlation_matrix


# =====================================================
# Data loading and generation functions
# =====================================================

def load_correlation_matrix_training(root_dir='',
                                     filename=''):
    """Load correlation matrix from training data"""
    file_path = os.path.join(root_path, root_dir, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training data file {filename} does not exist in path {root_dir}")

    correlation_matrix_training = np.load(file_path)
    print(f"✅ Successfully loaded training data, matrix shape: {correlation_matrix_training.shape}")
    return correlation_matrix_training


def generate_cma_optimized_matrix():
    """Generate correlation matrix from CMA optimized model"""
    print(f"\n🔄 Starting CMA optimized model generation...")
    print("-" * 60)

    # CMA optimization parameters JSON file path
    json_file_path = r""

    # Load optimization parameters
    print(f"Loading optimization parameters: {os.path.basename(json_file_path)}")
    with open(json_file_path, 'r') as json_file:
        parameters = json.load(json_file)

    # Extract parameters
    spam_rates = parameters.get('spam_rates', [])
    lp = parameters.get('lp', [])
    sp = parameters.get('sp', [])
    ecr_fid = parameters.get('ecr_fid', [])
    sqg_fid = parameters.get('sqg_fid', [])
    t1_t2_values = [(d['t1'], d['t2']) for d in parameters.get('t1_t2_values', [])]
    ecr_lengths = parameters.get('ecr_lengths', [])
    rd_lengths = parameters.get('rd_length', [])

    print(f"Parameters loaded:")
    print(f"  - SPAM rates: {len(spam_rates)} elements")
    print(f"  - Leakage parameters: {len(lp)} elements")
    print(f"  - T1/T2 values: {len(t1_t2_values)} elements")

    # Run simulation sampling
    print(f"Running simulation sampling...")
    print(f"  - SHOTS: {SHOTS}, SHOTS_EXP: {SHOTS_EXP}")
    print(f"  - NUM_ROUNDS: {NUM_ROUNDS}, NUM_QUBITS: {NUM_QUBITS}")

    results_array = run_sampling(SHOTS_EXP, SHOTS, NUM_ROUNDS - 1, NUM_QUBITS,
                                 lp, sp, spam_rates, sqg_fid, ecr_fid,
                                 t1_t2_values, ecr_lengths, rd_lengths)

    # Calculate correlation matrix
    print(f"Calculating correlation matrix...")
    M_sim = compute_correlation_matrix(results_array, NUM_QUBITS, NUM_ROUNDS, SHOTS * SHOTS_EXP)

    return M_sim


# Data loading and generation
print("=" * 80)
print("Loading and generating data...")
print("=" * 80)

# 1. Load training data
correlation_matrix_training = load_correlation_matrix_training()

# 2. Generate CMA optimized model data
correlation_matrix_sim_cma_op = generate_cma_optimized_matrix()


# =====================================================
# Correlation calculation functions
# =====================================================

def calculate_offset_sums(matrix, qubits, rounds_):
    """Calculate time correlations"""
    offset_sums = []
    for q in range(qubits):
        start_idx = q * rounds_
        end_idx = start_idx + rounds_
        one_offset_sum = 0
        for i in range(start_idx, end_idx - 1):
            one_offset_sum += matrix[i, i + 1]  # Above the diagonal
            one_offset_sum += matrix[i + 1, i]  # Below the diagonal
        offset_sums.append(one_offset_sum)
    return offset_sums


def calculate_inter_qubit_sums_no_symmetry(matrix, qubits, rounds_):
    """Calculate space correlations"""
    inter_qubit_sums = []
    for q in range(qubits - 1):
        start_idx_q1 = q * rounds_
        end_idx_q1 = start_idx_q1 + rounds_
        start_idx_q2 = (q + 1) * rounds_
        end_idx_q2 = start_idx_q2 + rounds_

        inter_qubit_sum = 0
        for i in range(rounds_):
            inter_qubit_sum += matrix[start_idx_q1 + i, start_idx_q2 + i]

        inter_qubit_sums.append(inter_qubit_sum)
    return inter_qubit_sums


def calculate_off_qubit_sums(matrix, qubits, rounds_):
    """Calculate spacetime correlations"""
    off_qubit_sums = []
    for q in range(qubits - 1):
        start_idx_q1 = q * rounds_
        end_idx_q1 = start_idx_q1 + rounds_
        start_idx_q2 = (q + 1) * rounds_
        end_idx_q2 = start_idx_q2 + rounds_

        off_qubit_sum = 0
        for i in range(rounds_ - 1):
            off_qubit_sum += matrix[start_idx_q1 + i, start_idx_q2 + i + 1]

        off_qubit_sums.append(off_qubit_sum)
    return off_qubit_sums


def calculate_difference_with_training(matrix):
    """Calculate difference with training dataset (returns detailed differences for each qubit)"""
    # Calculate training data correlations
    offset_diagonal_sums_train = calculate_offset_sums(correlation_matrix_training, NUM_MEA, NUM_ROUNDS)
    inter_qubit_sums_train = calculate_inter_qubit_sums_no_symmetry(correlation_matrix_training, NUM_MEA, NUM_ROUNDS)
    off_qubit_sums_train = calculate_off_qubit_sums(correlation_matrix_training, NUM_MEA, NUM_ROUNDS)

    # Calculate current matrix correlations
    offset_diagonal_sums_current = calculate_offset_sums(matrix, NUM_MEA, NUM_ROUNDS)
    inter_qubit_sums_current = calculate_inter_qubit_sums_no_symmetry(matrix, NUM_MEA, NUM_ROUNDS)
    off_qubit_sums_current = calculate_off_qubit_sums(matrix, NUM_MEA, NUM_ROUNDS)

    # Calculate absolute differences for each qubit
    time_diffs_per_qubit = np.abs(np.array(offset_diagonal_sums_current) - np.array(offset_diagonal_sums_train))
    space_diffs_per_qubit = np.abs(np.array(inter_qubit_sums_current) - np.array(inter_qubit_sums_train))
    spacetime_diffs_per_qubit = np.abs(np.array(off_qubit_sums_current) - np.array(off_qubit_sums_train))

    # Calculate total differences
    time_diff = np.sum(time_diffs_per_qubit)
    space_diff = np.sum(space_diffs_per_qubit)
    spacetime_diff = np.sum(spacetime_diffs_per_qubit)
    total_diff = time_diff + space_diff + spacetime_diff

    return {
        'time_diff': time_diff,
        'space_diff': space_diff,
        'spacetime_diff': spacetime_diff,
        'total_diff': total_diff,
        'time_diffs_per_qubit': time_diffs_per_qubit,
        'space_diffs_per_qubit': space_diffs_per_qubit,
        'spacetime_diffs_per_qubit': spacetime_diffs_per_qubit
    }


def extract_noise_probability(filename):
    """Extract noise probability from filename"""
    # Match scientific notation format like 1.000e-05
    match = re.search(r'(\d+\.\d+e-\d+)', filename)
    if match:
        return float(match.group(1))

    # Match regular decimal format like 0.01
    match = re.search(r'(\d+\.\d+)', filename)
    if match:
        return float(match.group(1))

    # Try to match other possible formats
    match = re.search(r'probs_(\d+\.\d+e-\d+)', filename)
    if match:
        return float(match.group(1))

    return None


# =====================================================
# Traditional noise model optimization functions
# =====================================================

def find_optimal_traditional_models():
    """Find optimal parameters for traditional noise models"""

    # Traditional noise model paths
    noise_models = {
        'rep_code_sim_circuit': 'Comparison/data/rep_code_sim_circuit',
        'rep_code_sim_code_capacity': 'Comparison/data/rep_code_sim_code_capacity',
        'rep_code_sim_phenomenological': 'Comparison/data/rep_code_sim_phenomenological',
        'rep_code_sim_SD6': 'Comparison/data/rep_code_sim_SD6',
        'rep_code_sim_SI1000': 'Comparison/data/rep_code_sim_SI1000'
    }

    results = {}

    for model_name, model_relative_path in noise_models.items():
        print(f"\nProcessing traditional noise model: {model_name}")
        print("-" * 60)

        model_path = os.path.join(root_path, model_relative_path)

        if not os.path.exists(model_path):
            print(f"Warning: Directory {model_path} does not exist")
            continue

        print(f"Search path: {model_path}")

        pattern = os.path.join(model_path, "*.npy")
        files = glob.glob(pattern)

        if not files:
            print(f"No .npy files found in {model_path}")
            continue

        print(f"Found {len(files)} .npy files")

        model_results = []

        for file_path in files:
            try:
                print(f"Processing: {os.path.basename(file_path)}")

                correlation_matrix = np.load(file_path)
                print(f"  Matrix shape: {correlation_matrix.shape}")

                diff_result = calculate_difference_with_training(correlation_matrix)
                noise_prob = extract_noise_probability(os.path.basename(file_path))

                result = {
                    'file_path': file_path,
                    'filename': os.path.basename(file_path),
                    'noise_probability': noise_prob,
                    'time_diff': diff_result['time_diff'],
                    'space_diff': diff_result['space_diff'],
                    'spacetime_diff': diff_result['spacetime_diff'],
                    'total_diff': diff_result['total_diff']
                }

                model_results.append(result)

                print(f"  Noise probability: {noise_prob}")
                print(f"  Time difference: {diff_result['time_diff']:.6f}")
                print(f"  Space difference: {diff_result['space_diff']:.6f}")
                print(f"  Spacetime difference: {diff_result['spacetime_diff']:.6f}")
                print(f"  Total difference: {diff_result['total_diff']:.6f}")
                print()

            except Exception as e:
                print(f"Error: Cannot process file {file_path}: {str(e)}")
                continue

        if model_results:
            optimal_result = min(model_results, key=lambda x: x['total_diff'])
            results[model_name] = {
                'optimal': optimal_result,
                'all_results': model_results
            }

            print(f"🏆 {model_name} optimal result:")
            print(f"  File: {optimal_result['filename']}")
            print(f"  Noise probability: {optimal_result['noise_probability']}")
            print(f"  Minimum total difference: {optimal_result['total_diff']:.6f}")

    return results


def get_optimized_model_result():
    """Get optimized model results"""
    print(f"\nProcessing optimized noise model: sim_cma_op")
    print("-" * 60)

    diff_result = calculate_difference_with_training(correlation_matrix_sim_cma_op)

    result = {
        'model_type': 'optimized',
        'filename': 'correlation_matrix_sim_cma_op_generated.npy',
        'noise_probability': None,  # Optimized model doesn't have a single noise probability
        'time_diff': diff_result['time_diff'],
        'space_diff': diff_result['space_diff'],
        'spacetime_diff': diff_result['spacetime_diff'],
        'total_diff': diff_result['total_diff'],
        'generation_method': 'generated'
    }

    print(f"  Time difference: {diff_result['time_diff']:.6f}")
    print(f"  Space difference: {diff_result['space_diff']:.6f}")
    print(f"  Spacetime difference: {diff_result['spacetime_diff']:.6f}")
    print(f"  Total difference: {diff_result['total_diff']:.6f}")
    print(f"  Generation method: Real-time generation")

    return {'sim_cma_op': result}


# =====================================================
# Visualization functions
# =====================================================

def plot_unified_comparison(traditional_results, optimized_results):
    """Plot unified comparison of difference breakdown between traditional and optimized models"""

    if not traditional_results and not optimized_results:
        print("No data available for plotting")
        return

    # Prepare data
    models = []
    time_diffs = []
    space_diffs = []
    spacetime_diffs = []

    # Add traditional model data
    for model, data in traditional_results.items():
        if 'optimal' in data and data['optimal']['noise_probability'] is not None:
            models.append(model.replace('rep_code_sim_', ''))
            time_diffs.append(data['optimal']['time_diff'])
            space_diffs.append(data['optimal']['space_diff'])
            spacetime_diffs.append(data['optimal']['spacetime_diff'])

    # Add optimized model data
    for model, data in optimized_results.items():
        models.append('CMA Optimized')
        time_diffs.append(data['time_diff'])
        space_diffs.append(data['space_diff'])
        spacetime_diffs.append(data['spacetime_diff'])

    if not models:
        print("No valid data available for plotting")
        return

    # Create chart
    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(models))
    width = 0.25

    bars1 = ax.bar(x - width, time_diffs, width, label='Time Difference', color='#e63946')
    bars2 = ax.bar(x, space_diffs, width, label='Space Difference', color='#2a9d8f')
    bars3 = ax.bar(x + width, spacetime_diffs, width, label='Spacetime Difference', color='#ffb703')

    ax.set_ylabel('Difference Value')
    ax.set_title('Difference Breakdown by Model (vs Training Data)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_per_qubit_differences(traditional_results, optimized_results):
    """Plot detailed per-qubit difference comparisons (three separate plots)"""

    # Prepare data
    models = []
    time_diffs_all = []
    space_diffs_all = []
    spacetime_diffs_all = []

    # Add traditional model data
    for model, data in traditional_results.items():
        if 'optimal' in data:
            models.append(model.replace('rep_code_sim_', ''))
            # Need to recalculate per-qubit differences
            matrix = np.load(data['optimal']['file_path'])
            diff_result = calculate_difference_with_training(matrix)
            time_diffs_all.append(diff_result['time_diffs_per_qubit'])
            space_diffs_all.append(diff_result['space_diffs_per_qubit'])
            spacetime_diffs_all.append(diff_result['spacetime_diffs_per_qubit'])

    # Add optimized model data
    for model, data in optimized_results.items():
        models.append('CMA Optimized')
        diff_result = calculate_difference_with_training(correlation_matrix_sim_cma_op)
        time_diffs_all.append(diff_result['time_diffs_per_qubit'])
        space_diffs_all.append(diff_result['space_diffs_per_qubit'])
        spacetime_diffs_all.append(diff_result['spacetime_diffs_per_qubit'])

    width = 0.8 / len(models)

    # Plot 1: Time differences (10 qubits)
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    x_time = np.arange(1, NUM_MEA + 1)

    for i, model in enumerate(models):
        offset = (i - len(models) / 2) * width + width / 2
        ax1.bar(x_time + offset, time_diffs_all[i], width,
                label=model, alpha=0.8, color=f'C{i}')

    ax1.set_xlabel('Qubit Index')
    ax1.set_ylabel('Time Difference')
    ax1.set_title('Time Difference by Qubit (vs Training Data)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot 2: Space differences (9 qubit pairs)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    x_space = np.arange(1, NUM_MEA)

    for i, model in enumerate(models):
        offset = (i - len(models) / 2) * width + width / 2
        ax2.bar(x_space + offset, space_diffs_all[i], width,
                label=model, alpha=0.8, color=f'C{i}')

    ax2.set_xlabel('Qubit Pair Index')
    ax2.set_ylabel('Space Difference')
    ax2.set_title('Space Difference by Qubit Pair (vs Training Data)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot 3: Spacetime differences (9 qubit pairs)
    fig3, ax3 = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(models):
        offset = (i - len(models) / 2) * width + width / 2
        ax3.bar(x_space + offset, spacetime_diffs_all[i], width,
                label=model, alpha=0.8, color=f'C{i}')

    ax3.set_xlabel('Qubit Pair Index')
    ax3.set_ylabel('Spacetime Difference')
    ax3.set_title('Spacetime Difference by Qubit Pair (vs Training Data)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_per_qubit_correlations(traditional_results, optimized_results):
    """Plot actual correlation value comparisons for each qubit (three separate plots)"""

    # Prepare data
    models = []
    time_corrs_all = []
    space_corrs_all = []
    spacetime_corrs_all = []

    # Add traditional model data
    for model, data in traditional_results.items():
        if 'optimal' in data:
            models.append(model.replace('rep_code_sim_', ''))
            # Load matrix and calculate actual correlation values
            matrix = np.load(data['optimal']['file_path'])
            time_corrs = calculate_offset_sums(matrix, NUM_MEA, NUM_ROUNDS)
            space_corrs = calculate_inter_qubit_sums_no_symmetry(matrix, NUM_MEA, NUM_ROUNDS)
            spacetime_corrs = calculate_off_qubit_sums(matrix, NUM_MEA, NUM_ROUNDS)

            time_corrs_all.append(time_corrs)
            space_corrs_all.append(space_corrs)
            spacetime_corrs_all.append(spacetime_corrs)

    # Add optimized model data
    for model, data in optimized_results.items():
        models.append('CMA Optimized')
        time_corrs = calculate_offset_sums(correlation_matrix_sim_cma_op, NUM_MEA, NUM_ROUNDS)
        space_corrs = calculate_inter_qubit_sums_no_symmetry(correlation_matrix_sim_cma_op, NUM_MEA, NUM_ROUNDS)
        spacetime_corrs = calculate_off_qubit_sums(correlation_matrix_sim_cma_op, NUM_MEA, NUM_ROUNDS)

        time_corrs_all.append(time_corrs)
        space_corrs_all.append(space_corrs)
        spacetime_corrs_all.append(spacetime_corrs)

    # Add Training data as reference
    models.append('Training')
    time_corrs_train = calculate_offset_sums(correlation_matrix_training, NUM_MEA, NUM_ROUNDS)
    space_corrs_train = calculate_inter_qubit_sums_no_symmetry(correlation_matrix_training, NUM_MEA, NUM_ROUNDS)
    spacetime_corrs_train = calculate_off_qubit_sums(correlation_matrix_training, NUM_MEA, NUM_ROUNDS)

    time_corrs_all.append(time_corrs_train)
    space_corrs_all.append(space_corrs_train)
    spacetime_corrs_all.append(spacetime_corrs_train)

    width = 0.8 / len(models)

    # Plot 1: Time correlations (10 qubits)
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    x_time = np.arange(1, NUM_MEA + 1)

    for i, model in enumerate(models):
        offset = (i - len(models) / 2) * width + width / 2
        ax1.bar(x_time + offset, time_corrs_all[i], width,
                label=model, alpha=0.8, color=f'C{i}')

    ax1.set_xlabel('Qubit Index')
    ax1.set_ylabel('Sum of One-Offset Diagonal Elements')
    ax1.set_title('Comparison of Time Correlations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot 2: Space correlations (9 qubit pairs)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    x_space = np.arange(1, NUM_MEA)

    for i, model in enumerate(models):
        offset = (i - len(models) / 2) * width + width / 2
        ax2.bar(x_space + offset, space_corrs_all[i], width,
                label=model, alpha=0.8, color=f'C{i}')

    ax2.set_xlabel('Qubit Pair Index')
    ax2.set_ylabel('Sum of Inter-Qubit Diagonal Elements')
    ax2.set_title('Comparison of Space Correlations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot 3: Spacetime correlations (9 qubit pairs)
    fig3, ax3 = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(models):
        offset = (i - len(models) / 2) * width + width / 2
        ax3.bar(x_space + offset, spacetime_corrs_all[i], width,
                label=model, alpha=0.8, color=f'C{i}')

    ax3.set_xlabel('Qubit Pair Index')
    ax3.set_ylabel('Sum of Off-Diagonal Elements')
    ax3.set_title('Comparison of Spacetime Correlations')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def print_unified_summary_table(traditional_results, optimized_results):
    """Print unified summary table"""

    # Collect all results
    all_results = []

    # Add traditional model results
    for model, data in traditional_results.items():
        if 'optimal' in data:
            opt = data['optimal']
            all_results.append({
                'model_name': model.replace('rep_code_sim_', ''),
                'model_type': 'Traditional',
                'noise_probability': opt['noise_probability'],
                'time_diff': opt['time_diff'],
                'space_diff': opt['space_diff'],
                'spacetime_diff': opt['spacetime_diff'],
                'total_diff': opt['total_diff']
            })

    # Add optimized model results
    for model, data in optimized_results.items():
        all_results.append({
            'model_name': 'CMA Optimized',
            'model_type': 'Optimized',
            'noise_probability': None,
            'time_diff': data['time_diff'],
            'space_diff': data['space_diff'],
            'spacetime_diff': data['spacetime_diff'],
            'total_diff': data['total_diff']
        })

    if not all_results:
        print("No results to display")
        return

    print("\n" + "=" * 110)
    print("Unified Comparison Results with Training Data")
    print("=" * 110)
    print(f"{'Model Name':<20} {'Type':<12} {'Optimal Noise Prob':<15} {'Time Diff':<12} {'Space Diff':<12} {'Spacetime Diff':<12} {'Total Diff':<12}")
    print("-" * 110)

    # Sort by total difference
    sorted_results = sorted(all_results, key=lambda x: x['total_diff'])

    for result in sorted_results:
        noise_prob_str = f"{result['noise_probability']:.2e}" if result['noise_probability'] is not None else "N/A"
        print(f"{result['model_name']:<20} {result['model_type']:<12} {noise_prob_str:<15} "
              f"{result['time_diff']:<12.6f} {result['space_diff']:<12.6f} "
              f"{result['spacetime_diff']:<12.6f} {result['total_diff']:<12.6f}")

    print("=" * 110)
    print("Note: Sorted by total difference (ascending), smaller values indicate better agreement with training data")
    print("CMA Optimized model does not have a single noise probability parameter")


def save_unified_results(traditional_results, optimized_results):
    """Save data for 7 plots to JSON file"""

    plot_data = {
        'metadata': {
            'description': 'Data for 7 plots: 1 difference breakdown + 3 per-qubit differences + 3 per-qubit correlations',
            'num_qubits': NUM_MEA,
            'num_rounds': NUM_ROUNDS
        },
        'plot1_difference_breakdown': {
            'description': 'Overall difference breakdown by model (vs Training Data)',
            'models': [],
            'time_diffs': [],
            'space_diffs': [],
            'spacetime_diffs': []
        },
        'plot2_time_differences_per_qubit': {
            'description': 'Time differences by qubit (vs Training Data)',
            'qubit_indices': list(range(1, NUM_MEA + 1)),
            'models': {}
        },
        'plot3_space_differences_per_qubit': {
            'description': 'Space differences by qubit pair (vs Training Data)',
            'qubit_pair_indices': list(range(1, NUM_MEA)),
            'models': {}
        },
        'plot4_spacetime_differences_per_qubit': {
            'description': 'Spacetime differences by qubit pair (vs Training Data)',
            'qubit_pair_indices': list(range(1, NUM_MEA)),
            'models': {}
        },
        'plot5_time_correlations': {
            'description': 'Actual time correlations by qubit',
            'qubit_indices': list(range(1, NUM_MEA + 1)),
            'models': {}
        },
        'plot6_space_correlations': {
            'description': 'Actual space correlations by qubit pair',
            'qubit_pair_indices': list(range(1, NUM_MEA)),
            'models': {}
        },
        'plot7_spacetime_correlations': {
            'description': 'Actual spacetime correlations by qubit pair',
            'qubit_pair_indices': list(range(1, NUM_MEA)),
            'models': {}
        }
    }

    # Collect data for plot 1: Overall difference breakdown
    for model, data in traditional_results.items():
        if 'optimal' in data:
            model_name = model.replace('rep_code_sim_', '')
            plot_data['plot1_difference_breakdown']['models'].append(model_name)
            plot_data['plot1_difference_breakdown']['time_diffs'].append(data['optimal']['time_diff'])
            plot_data['plot1_difference_breakdown']['space_diffs'].append(data['optimal']['space_diff'])
            plot_data['plot1_difference_breakdown']['spacetime_diffs'].append(data['optimal']['spacetime_diff'])

    for model, data in optimized_results.items():
        plot_data['plot1_difference_breakdown']['models'].append('CMA_Optimized')
        plot_data['plot1_difference_breakdown']['time_diffs'].append(data['time_diff'])
        plot_data['plot1_difference_breakdown']['space_diffs'].append(data['space_diff'])
        plot_data['plot1_difference_breakdown']['spacetime_diffs'].append(data['spacetime_diff'])

    # Collect data for plots 2-4: Per-qubit differences
    for model, data in traditional_results.items():
        if 'optimal' in data:
            model_name = model.replace('rep_code_sim_', '')
            matrix = np.load(data['optimal']['file_path'])
            diff_result = calculate_difference_with_training(matrix)

            plot_data['plot2_time_differences_per_qubit']['models'][model_name] = diff_result[
                'time_diffs_per_qubit'].tolist()
            plot_data['plot3_space_differences_per_qubit']['models'][model_name] = diff_result[
                'space_diffs_per_qubit'].tolist()
            plot_data['plot4_spacetime_differences_per_qubit']['models'][model_name] = diff_result[
                'spacetime_diffs_per_qubit'].tolist()

    for model, data in optimized_results.items():
        diff_result = calculate_difference_with_training(correlation_matrix_sim_cma_op)

        plot_data['plot2_time_differences_per_qubit']['models']['CMA_Optimized'] = diff_result[
            'time_diffs_per_qubit'].tolist()
        plot_data['plot3_space_differences_per_qubit']['models']['CMA_Optimized'] = diff_result[
            'space_diffs_per_qubit'].tolist()
        plot_data['plot4_spacetime_differences_per_qubit']['models']['CMA_Optimized'] = diff_result[
            'spacetime_diffs_per_qubit'].tolist()

    # Collect data for plots 5-7: Per-qubit actual correlations
    for model, data in traditional_results.items():
        if 'optimal' in data:
            model_name = model.replace('rep_code_sim_', '')
            matrix = np.load(data['optimal']['file_path'])

            time_corrs = calculate_offset_sums(matrix, NUM_MEA, NUM_ROUNDS)
            space_corrs = calculate_inter_qubit_sums_no_symmetry(matrix, NUM_MEA, NUM_ROUNDS)
            spacetime_corrs = calculate_off_qubit_sums(matrix, NUM_MEA, NUM_ROUNDS)

            plot_data['plot5_time_correlations']['models'][model_name] = time_corrs
            plot_data['plot6_space_correlations']['models'][model_name] = space_corrs
            plot_data['plot7_spacetime_correlations']['models'][model_name] = spacetime_corrs

    for model, data in optimized_results.items():
        time_corrs = calculate_offset_sums(correlation_matrix_sim_cma_op, NUM_MEA, NUM_ROUNDS)
        space_corrs = calculate_inter_qubit_sums_no_symmetry(correlation_matrix_sim_cma_op, NUM_MEA, NUM_ROUNDS)
        spacetime_corrs = calculate_off_qubit_sums(correlation_matrix_sim_cma_op, NUM_MEA, NUM_ROUNDS)

        plot_data['plot5_time_correlations']['models']['CMA_Optimized'] = time_corrs
        plot_data['plot6_space_correlations']['models']['CMA_Optimized'] = space_corrs
        plot_data['plot7_spacetime_correlations']['models']['CMA_Optimized'] = spacetime_corrs

    # Add Training data to plots 5-7 (as reference)
    time_corrs_train = calculate_offset_sums(correlation_matrix_training, NUM_MEA, NUM_ROUNDS)
    space_corrs_train = calculate_inter_qubit_sums_no_symmetry(correlation_matrix_training, NUM_MEA, NUM_ROUNDS)
    spacetime_corrs_train = calculate_off_qubit_sums(correlation_matrix_training, NUM_MEA, NUM_ROUNDS)

    plot_data['plot5_time_correlations']['models']['Training'] = time_corrs_train
    plot_data['plot6_space_correlations']['models']['Training'] = space_corrs_train
    plot_data['plot7_spacetime_correlations']['models']['Training'] = spacetime_corrs_train

    # Save to file
    filename = ''
    try:
        with open(filename, 'w') as f:
            json.dump(plot_data, f, indent=2)

        print(f"\n✅ Data for 7 plots saved to '{filename}' file")
        print(f"   Including:")
        print(f"   - Plot 1: Overall difference breakdown")
        print(f"   - Plot 2-4: Detailed per-qubit difference data")
        print(f"   - Plot 5-7: Actual per-qubit correlation data")

    except Exception as e:
        print(f"\n⚠️ Error saving results: {e}")


# =====================================================
# Main function
# =====================================================

def main():
    """Main function"""
    print("Starting to find optimal parameters for traditional noise models and compare with optimized models...")

    print("=" * 80)

    # 1. Find optimal parameters for traditional models
    print("Step 1: Finding optimal parameters for traditional noise models")
    traditional_results = find_optimal_traditional_models()

    # 2. Get optimized model results
    print("\nStep 2: Getting optimized noise model results")
    optimized_results = get_optimized_model_result()

    if not traditional_results and not optimized_results:
        print("\n❌ No valid results found")
        return None

    print("=" * 80)

    # 3. Print unified summary table
    print_unified_summary_table(traditional_results, optimized_results)

    # 4. Plot unified comparison charts
    # Plot difference breakdown comparison
    plot_unified_comparison(traditional_results, optimized_results)

    # Plot detailed per-qubit difference comparisons
    plot_per_qubit_differences(traditional_results, optimized_results)

    # Plot actual per-qubit correlation value comparisons
    plot_per_qubit_correlations(traditional_results, optimized_results)

    # 5. Save unified results to JSON file
    save_unified_results(traditional_results, optimized_results)

    return traditional_results, optimized_results


if __name__ == "__main__":
    results = main()