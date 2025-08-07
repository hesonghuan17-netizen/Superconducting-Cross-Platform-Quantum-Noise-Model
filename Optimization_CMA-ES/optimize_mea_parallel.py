import numpy as np
import cma
import sys
import os
import json
import time
import random

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from Simulation.Creation_and_Sampling.sampling import run_sampling
from config import NUM_QUBITS, NUM_ROUNDS, SHOTS, SHOTS_EXP, NUM_MEA

# ==================== File Path Configuration ====================
# Read path configuration from environment variables
shared_json_file = os.environ.get('SHARED_PARAMS_FILE')
json_file_path_train = os.environ.get('TRAINING_DATA_FILE')
correlation_matrix_training = np.load(json_file_path_train)

sigma = float(os.environ.get('CMA_MEA_SIGMA'))
popsize = int(os.environ.get('CMA_MEA_POPSIZE'))
maxiter = int(os.environ.get('CMA_MEA_MAXITER'))
tolfun = float(os.environ.get('CMA_MEA_TOLFUN'))
tolx = float(os.environ.get('CMA_MEA_TOLX'))

# Output log file
generation_log = os.environ.get('MEA_LOG_FILE')


# ==================== Windows-compatible file read/write functions ====================
def safe_read_json(file_path, max_retries=10, base_delay=0.1):
    """Windows-compatible safe JSON file reading"""
    for attempt in range(max_retries):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, IOError, PermissionError) as e:
            if attempt < max_retries - 1:
                # Exponential backoff + random delay
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                print(f"JSON read failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(delay)
                continue
            else:
                raise e
    return None


def safe_write_json(file_path, data, max_retries=10, base_delay=0.1):
    """Windows-compatible safe JSON file writing"""
    for attempt in range(max_retries):
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Use temporary file + atomic replacement
            temp_file = file_path + f'.tmp_{os.getpid()}_{int(time.time())}'

            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk

            # Atomic replacement on Windows
            if os.path.exists(file_path):
                backup_file = file_path + '.backup'
                os.replace(file_path, backup_file)  # Backup original file
                os.replace(temp_file, file_path)  # Replace with new file
                if os.path.exists(backup_file):
                    os.remove(backup_file)  # Delete backup
            else:
                os.replace(temp_file, file_path)

            return True

        except (IOError, OSError, PermissionError) as e:
            # Clean up temporary file
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                print(f"JSON write failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(delay)
                continue
            else:
                raise e
    return False


def get_latest_parameters():
    """Get latest parameters"""
    try:
        return safe_read_json(shared_json_file)
    except Exception as e:
        print(f"Measurement optimizer cannot read shared parameter file: {e}")
        raise e


def update_mea_parameters(new_spam_rates):
    """Update measurement error parameters in shared parameter file"""
    try:
        current_params = safe_read_json(shared_json_file)
        current_params['spam_rates'] = new_spam_rates
        safe_write_json(shared_json_file, current_params)
        return True
    except Exception as e:
        print(f"Measurement optimizer failed to update shared parameters: {e}")
        return False


# ==================== Optimization-related functions ====================
function_evaluation_counter = 0


def calculate_time_correlations(matrix, num_mea, rounds_):
    """Calculate time correlations (for each measurement qubit)"""
    time_correlations = []
    for q in range(num_mea):
        start_idx = q * rounds_
        end_idx = start_idx + rounds_
        time_corr = 0
        for i in range(start_idx, end_idx - 1):
            time_corr += matrix[i, i + 1]  # Upper diagonal
            time_corr += matrix[i + 1, i]  # Lower diagonal
        time_correlations.append(time_corr)
    return np.array(time_correlations)

def compute_correlation_matrix(data, num_qubits, rounds_, shots):
    """
    Computes the correlation matrix from quantum measurement data.

    Parameters:
    - data : np.array
        The results array from quantum measurements.
    - num_qubits : int
        Total number of qubits.
    - rounds : int
        Number of measurement rounds.
    - shots : int
        Total number of measurement shots.

    Returns:
    - np.array
        The computed correlation matrix.
    """
    num_mea = (num_qubits - 1) // 2
    # Reshape the data to (shots, rounds, qubits)
    data1 = data.reshape(shots, rounds_, num_mea)

    # Transpose data to switch 'rounds' and 'qubits' to make qubits sequential per shot
    data2 = data1.transpose(0, 2, 1)

    # Flatten the data back to (shots, qubits * rounds)
    flattened_data = data2.reshape(shots, num_mea * rounds_)

    # Compute probabilities
    p_x = np.mean(flattened_data, axis=0)
    p_xi_xj = np.zeros((num_mea * rounds_, num_mea * rounds_))

    # Efficient computation of p_xi_xj using vectorized operations
    for k in range(shots):
        temp = np.outer(flattened_data[k], flattened_data[k])
        p_xi_xj += temp

    p_xi_xj /= shots

    # Initialize correlation matrix
    correlation_matrix = np.zeros((num_mea * rounds_, num_mea * rounds_))

    # Calculate correlations efficiently
    xi = p_x.reshape(-1, 1)
    xj = p_x.reshape(1, -1)

    xi_xj = p_xi_xj

    term1 = (1 - 2 * xi)
    term2 = (1 - 2 * xj)

    # Avoid division by zero by checking where terms are non-zero
    valid_indices = np.where((term1 != 0) & (term2 != 0))

    # Compute correlation matrix for valid indices using correct array indexing
    for i, j in zip(*valid_indices):
        if i != j:
            correlation_matrix[i, j] = (xi_xj[i, j] - xi[i, 0] * xj[0, j]) / (term1[i, 0] * term2[0, j])

    # Since the correlation matrix is symmetric, mirror the results
    np.fill_diagonal(correlation_matrix, 0)
    correlation_matrix_sim = np.triu(correlation_matrix) + np.triu(correlation_matrix, 1).T

    return correlation_matrix_sim

def objective_function(mea_params):
    """Objective function - focused on time correlations"""
    global function_evaluation_counter
    function_evaluation_counter += 1

    # Get latest parameters
    current_params = get_latest_parameters()

    # Build complete spam_rates (update measurement error part)
    spam_rates = current_params.get('spam_rates', []).copy()

    # Extract measurement error indices (odd indices)
    mea_indices = [i for i in range(1, len(spam_rates), 2)]

    # Update measurement error parameters
    for idx, mea_param in zip(mea_indices, mea_params):
        spam_rates[idx] = mea_param

    # Extract parameters that may have been updated by other optimizers
    lp = current_params.get('lp', [])
    sp = current_params.get('sp', [])
    sqg_fid = current_params.get('sqg_fid', [])
    ecr_fid = current_params.get('ecr_fid', [])
    t1_t2_values = [(d['t1'], d['t2']) for d in current_params.get('t1_t2_values', [])]
    ecr_lengths = current_params.get('ecr_lengths', [])
    rd_lengths = current_params.get('rd_length', [])
    sqg_length_list = current_params.get('sqg_length', [])
    sqg_lengths = sqg_length_list[0] if sqg_length_list else 0

    # Run simulation
    results_array = run_sampling(SHOTS_EXP, SHOTS, NUM_ROUNDS - 1, NUM_QUBITS,
                                 lp, sp, spam_rates, sqg_fid, ecr_fid,
                                 t1_t2_values, ecr_lengths, rd_lengths, sqg_lengths)

    # Calculate correlation matrix
    M_sim = compute_correlation_matrix(results_array, NUM_QUBITS, NUM_ROUNDS, SHOTS * SHOTS_EXP)
    M_exp = correlation_matrix_training

    # Calculate time correlations
    time_corr_sim = calculate_time_correlations(M_sim, NUM_MEA, NUM_ROUNDS)
    time_corr_exp = calculate_time_correlations(M_exp, NUM_MEA, NUM_ROUNDS)

    # Calculate differences for each qubit
    differences = np.abs(time_corr_sim - time_corr_exp)
    objective_value = np.sum(differences)

    print(f"Measurement Eval {function_evaluation_counter:4d}: {objective_value:.6f}")
    print(f"  Per-qubit differences: {differences}")
    print(f"  Measurement error range: [{np.min(mea_params):.2e}, {np.max(mea_params):.2e}]")
    sys.stdout.flush()

    return objective_value


# ==================== Initialize shared parameter file ====================
initial_json_file = os.environ.get('INITIAL_PARAMS_FILE')

if not os.path.exists(shared_json_file):
    print("Measurement optimizer: Shared parameter file does not exist, creating...")
    try:
        with open(initial_json_file, 'r') as f:
            initial_params = json.load(f)
        safe_write_json(shared_json_file, initial_params)
        print(f"Measurement optimizer: Shared parameter file created: {shared_json_file}")
    except Exception as e:
        print(f"Measurement optimizer: Failed to create shared parameter file: {e}")
        sys.exit(1)

# Get initial parameters
initial_params = get_latest_parameters()
initial_spam_rates = initial_params.get('spam_rates', [])

# Extract measurement error parameters (odd indices)
mea_indices = [i for i in range(1, len(initial_spam_rates), 2)]
initial_mea_rates = [initial_spam_rates[i] for i in mea_indices]

print(f"Measurement optimizer started")
print(f"Initial measurement errors: {initial_mea_rates}")
print(f"Measurement error indices: {mea_indices}")
print(f"Optimization parameter count: {len(initial_mea_rates)}")

# ==================== Optimization setup and execution ====================
initial_mea_params = np.array(initial_mea_rates)

# Measurement error parameter bounds
mea_lower_bound = 1e-4
mea_upper_bound = 2e-1
lower_bounds = [mea_lower_bound] * NUM_MEA
upper_bounds = [mea_upper_bound] * NUM_MEA

# Create generations file
with open(generation_log, "w") as f:
    f.write("Generation,BestObjective,BestParameters\n")

# Use dynamic configuration
generation_counter = 0


opts = {
    'bounds': [lower_bounds, upper_bounds],
    'maxiter': maxiter,
    'popsize': popsize,
    'verb_log': 0,
    'verb_disp': 0,
    'tolfun': tolfun,
    'tolx': tolx,
    'seed': 42,
}

print("Measurement optimizer: Starting CMA-ES optimization...")
print(f"Population size: {popsize}")
print(f"Max generations: {opts['maxiter']}")
print(f"Parameter bounds: [{mea_lower_bound:.1e}, {mea_upper_bound:.1e}]")
print(f"Measurement parameter count: {NUM_MEA}")

es = cma.CMAEvolutionStrategy(initial_mea_params, sigma, opts)

# Optimization loop
while not es.stop():
    generation_counter += 1
    print(f"\nMeasurement Generation {generation_counter}")

    solutions = es.ask()
    fitness_values = []

    for i, solution in enumerate(solutions):
        fitness = objective_function(solution)
        fitness_values.append(fitness)

    es.tell(solutions, fitness_values)

    current_best = es.best.f if es.best.f is not None else min(fitness_values)
    best_solution = es.best.x if es.best.x is not None else solutions[np.argmin(fitness_values)]

    # Record to generations file
    with open(generation_log, "a") as f:
        param_str = ",".join(map(str, best_solution))
        f.write(f"{generation_counter},{current_best},{param_str}\n")

    print(f"Measurement Generation {generation_counter}: Best = {current_best:.6f}")

    # Update shared parameter file every 10 generations
    if generation_counter % 10 == 0:
        # Build complete spam_rates
        current_params = get_latest_parameters()
        updated_spam_rates = current_params.get('spam_rates', []).copy()

        # Update measurement error part
        for idx, mea_param in zip(mea_indices, best_solution):
            updated_spam_rates[idx] = mea_param

        success = update_mea_parameters(updated_spam_rates)
        if success:
            print(f"  >>> Measurement error parameters updated to shared file (generation {generation_counter}) <<<")
        else:
            print(f"  >>> Measurement error parameter update failed (generation {generation_counter}) <<<")

# ==================== Save final results ====================
final_solution = es.best.x
final_fitness = es.best.f

# Build final spam_rates
final_params = get_latest_parameters()
final_spam_rates = final_params.get('spam_rates', []).copy()

# Update measurement error part
for idx, mea_param in zip(mea_indices, final_solution):
    final_spam_rates[idx] = mea_param

# Update final parameters to shared file
success = update_mea_parameters(final_spam_rates)

print(f"\nMeasurement optimization completed!")
print(f"Final objective value: {final_fitness:.6f}")
print(f"Total generations: {generation_counter}")
print(f"Total function evaluations: {function_evaluation_counter}")
print(f"Final measurement errors: {final_solution}")

print(f"\nMeasurement error analysis:")
for i, mea_val in enumerate(final_solution):
    print(f"Qubit {i} measurement error: {mea_val:.2e}")

print(f"Measurement error range: [{np.min(final_solution):.2e}, {np.max(final_solution):.2e}]")

if success:
    print(f"Final measurement error parameters saved to shared file: {shared_json_file}")
else:
    print(f"Final measurement error parameter save failed")

print(f"Output files:")
print(f"- Shared JSON: {shared_json_file}")
print(f"- Generations: {generation_log}")