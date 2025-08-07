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
shared_json_file = os.environ.get('SHARED_PARAMS_FILE')
json_file_path_train = os.environ.get('TRAINING_DATA_FILE')
generation_log = os.environ.get('T1T2_LOG_FILE')

sigma = float(os.environ.get('CMA_T1T2_SIGMA'))
popsize = int(os.environ.get('CMA_T1T2_POPSIZE'))
maxiter = int(os.environ.get('CMA_T1T2_MAXITER'))
tolfun = float(os.environ.get('CMA_T1T2_TOLFUN'))
tolx = float(os.environ.get('CMA_T1T2_TOLX'))


# Load training data
correlation_matrix_training = np.load(json_file_path_train)


# ==================== Windows-compatible file read/write functions ====================
def safe_read_json(file_path, max_retries=5, base_delay=0.1):
    """Windows-compatible safe JSON file reading"""
    for attempt in range(max_retries):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, IOError, PermissionError) as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                print(f"T1T2 optimizer JSON read failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(delay)
                continue
            else:
                raise e


def safe_write_json(file_path, data, max_retries=5, base_delay=0.1):
    """Windows-compatible safe JSON file writing"""
    for attempt in range(max_retries):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            temp_file = file_path + f'.tmp_{os.getpid()}_{int(time.time())}'

            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())

            if os.path.exists(file_path):
                backup_file = file_path + '.backup'
                os.replace(file_path, backup_file)
                os.replace(temp_file, file_path)
                if os.path.exists(backup_file):
                    os.remove(backup_file)
            else:
                os.replace(temp_file, file_path)

            return True

        except (IOError, OSError, PermissionError) as e:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                print(f"T1T2 optimizer JSON write failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(delay)
                continue
            else:
                print(f"T1T2 optimizer JSON write finally failed, skipping")
                return False


def get_latest_parameters():
    """Get latest parameters"""
    return safe_read_json(shared_json_file)


def update_t1t2_parameters(new_t1t2_values):
    """Update T1/T2 parameters in shared parameter file"""
    try:
        current_params = safe_read_json(shared_json_file)
        current_params['t1_t2_values'] = new_t1t2_values
        return safe_write_json(shared_json_file, current_params)
    except Exception as e:
        print(f"T1T2 optimizer failed to update shared parameters: {e}")
        return False


# ==================== T1/T2 parameter transformation functions ====================
# T1/T2 physical bounds
t1_lower_bound = 1e-5  # 10 microseconds
t1_upper_bound = 6e-4  # 600 microseconds
ratio_lower_bound = 0.1  # T2/T1 minimum ratio
ratio_upper_bound = 1.0  # T2/T1 maximum ratio


def transform_to_physical(normalized_params):
    """Convert normalized parameters [0,1] to physical T1/T2 values"""
    log_t1_norm = normalized_params[:NUM_QUBITS]
    log_ratio_norm = normalized_params[NUM_QUBITS:]

    # Transform log(T1): [0,1] → [log10(1e-5), log10(6e-4)]
    log_t1_real = log_t1_norm * (np.log10(t1_upper_bound) - np.log10(t1_lower_bound)) + np.log10(t1_lower_bound)
    t1_params = 10 ** log_t1_real

    # Transform log(T2/T1): [0,1] → [log10(0.1), log10(1.0)]
    log_ratio_real = log_ratio_norm * (np.log10(ratio_upper_bound) - np.log10(ratio_lower_bound)) + np.log10(
        ratio_lower_bound)
    ratio_params = 10 ** log_ratio_real

    # Calculate T2 = T1 * ratio (ensure T2 ≤ T1)
    t2_params = t1_params * ratio_params

    return t1_params, t2_params


def transform_to_normalized(t1_params, t2_params):
    """Convert physical T1/T2 values to normalized parameters [0,1]"""
    # Calculate T2/T1 ratio and limit within bounds
    ratios = np.clip(t2_params / t1_params, ratio_lower_bound, ratio_upper_bound)
    t1_clipped = np.clip(t1_params, t1_lower_bound, t1_upper_bound)

    # Reverse transformation
    log_t1_real = np.log10(t1_clipped)
    log_t1_norm = (log_t1_real - np.log10(t1_lower_bound)) / (np.log10(t1_upper_bound) - np.log10(t1_lower_bound))

    log_ratio_real = np.log10(ratios)
    log_ratio_norm = (log_ratio_real - np.log10(ratio_lower_bound)) / (
                np.log10(ratio_upper_bound) - np.log10(ratio_lower_bound))

    return np.concatenate([log_t1_norm, log_ratio_norm])


# ==================== Optimization-related functions ====================
function_evaluation_counter = 0


def calculate_space_correlations(matrix, num_mea, rounds_):
    """Calculate space correlations (between adjacent qubits)"""
    space_correlations = []
    for q in range(num_mea - 1):
        start_idx_q1 = q * rounds_
        start_idx_q2 = (q + 1) * rounds_
        space_corr = sum(matrix[start_idx_q1 + i, start_idx_q2 + i] for i in range(rounds_))
        space_correlations.append(space_corr)
    return np.array(space_correlations)


def calculate_time_correlations(matrix, num_mea, rounds_):
    """Calculate time correlations (within each qubit)"""
    time_correlations = []
    for q in range(num_mea):
        start_idx = q * rounds_
        end_idx = start_idx + rounds_
        time_corr = sum(matrix[i, i + 1] + matrix[i + 1, i] for i in range(start_idx, end_idx - 1))
        time_correlations.append(time_corr)
    return np.array(time_correlations)


def calculate_spacetime_correlations(matrix, qubits, rounds_):
    """Calculate spacetime correlations"""
    spacetime_correlations = []
    for q in range(qubits - 1):
        start_idx_q1 = q * rounds_
        start_idx_q2 = (q + 1) * rounds_
        spacetime_corr = sum(matrix[start_idx_q1 + i, start_idx_q2 + i + 1] for i in range(rounds_ - 1))
        spacetime_correlations.append(spacetime_corr)
    return np.array(spacetime_correlations)

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

def objective_function(normalized_params):
    """Comprehensive objective function"""
    global function_evaluation_counter
    function_evaluation_counter += 1

    # Convert to physical parameters
    t1_params, t2_params = transform_to_physical(normalized_params)

    # Basic validation
    if np.any(t1_params <= 0) or np.any(t2_params <= 0):
        return 1e6
    if np.any(t2_params > t1_params * 1.01):  # Allow small numerical error
        return 1e6

    # Get latest parameters
    current_params = get_latest_parameters()

    # Create t1_t2_values list
    t1t2_values = [(float(t1), float(t2)) for t1, t2 in zip(t1_params, t2_params)]

    # Extract other parameters
    lp = current_params.get('lp', [])
    sp = current_params.get('sp', [])
    spam_rates = current_params.get('spam_rates', [])
    sqg_fid = current_params.get('sqg_fid', [])
    ecr_fid = current_params.get('ecr_fid', [])
    ecr_lengths = current_params.get('ecr_lengths', [])
    rd_lengths = current_params.get('rd_length', [])
    sqg_length_list = current_params.get('sqg_length', [])
    sqg_lengths = sqg_length_list[0] if sqg_length_list else 0

    # Run simulation
    results_array = run_sampling(
        SHOTS_EXP, SHOTS, NUM_ROUNDS - 1, NUM_QUBITS,
        lp, sp, spam_rates, sqg_fid, ecr_fid,
        t1t2_values, ecr_lengths, rd_lengths, sqg_lengths
    )

    # Calculate correlation matrix
    M_sim = compute_correlation_matrix(results_array, NUM_QUBITS, NUM_ROUNDS, SHOTS * SHOTS_EXP)
    M_exp = correlation_matrix_training

    # Calculate three types of correlations
    spatial_corr_sim = calculate_space_correlations(M_sim, NUM_MEA, NUM_ROUNDS)
    spatial_corr_exp = calculate_space_correlations(M_exp, NUM_MEA, NUM_ROUNDS)
    spatial_diff = np.sum(np.abs(spatial_corr_sim - spatial_corr_exp))

    temporal_corr_sim = calculate_time_correlations(M_sim, NUM_MEA, NUM_ROUNDS)
    temporal_corr_exp = calculate_time_correlations(M_exp, NUM_MEA, NUM_ROUNDS)
    temporal_diff = np.sum(np.abs(temporal_corr_sim - temporal_corr_exp))

    spacetime_corr_sim = calculate_spacetime_correlations(M_sim, NUM_MEA, NUM_ROUNDS)
    spacetime_corr_exp = calculate_spacetime_correlations(M_exp, NUM_MEA, NUM_ROUNDS)
    spacetime_diff = np.sum(np.abs(spacetime_corr_sim - spacetime_corr_exp))

    # Comprehensive objective function
    objective_value = 0.6 * spatial_diff + 0.1 * temporal_diff + 0.3 * spacetime_diff

    print(f"T1T2 Eval {function_evaluation_counter:4d}: {objective_value:.6f}")
    sys.stdout.flush()

    return objective_value


# ==================== Initialization ====================
# Initialize shared parameter file (if it doesn't exist)
initial_json_file = os.environ.get('INITIAL_PARAMS_FILE')
if not os.path.exists(shared_json_file) and initial_json_file:
    print("T1T2 optimizer: Initializing shared parameter file...")
    try:
        with open(initial_json_file, 'r') as f:
            initial_params = json.load(f)
        safe_write_json(shared_json_file, initial_params)
        print("T1T2 optimizer: Shared parameter file created successfully")
    except Exception as e:
        print(f"T1T2 optimizer: Failed to create shared parameter file: {e}")
        sys.exit(1)

# Get initial T1/T2 parameters
initial_params = get_latest_parameters()
initial_t1_t2_values = initial_params.get('t1_t2_values', [])
initial_t1 = np.array([val['t1'] for val in initial_t1_t2_values])
initial_t2 = np.array([val['t2'] for val in initial_t1_t2_values])

print(f"Initial T1 values: {initial_t1}")
print(f"Initial T2 values: {initial_t2}")

# Convert to normalized space
initial_normalized_params = transform_to_normalized(initial_t1, initial_t2)
print(f"Normalized parameter conversion completed")

# ==================== Optimization setup and execution ====================
total_params = 2 * NUM_QUBITS

# Create generations file, record physical T1/T2 original values
with open(generation_log, "w") as f:
    # Build detailed column headers
    headers = ["Generation", "BestObjective"]
    for i in range(NUM_QUBITS):
        headers.append(f"T1_qubit_{i}")
        headers.append(f"T2_qubit_{i}")
    f.write(",".join(headers) + "\n")


generation_counter = 0


opts = {
    'bounds': [[0.0] * total_params, [1.0] * total_params],
    'maxiter': maxiter,
    'popsize': popsize,
    'verb_log': 0,
    'verb_disp': 0,
    'tolfun': tolfun,
    'tolx': tolx,
    'seed': 42,
}

print("T1T2 optimizer: Starting CMA-ES optimization...")
print(f"Population size: {popsize}, Max generations: {opts['maxiter']}")
print(f"Total parameters: {total_params}")
print(f"T1 bounds: [{t1_lower_bound:.1e}, {t1_upper_bound:.1e}]")
print(f"T2/T1 ratio bounds: [{ratio_lower_bound}, {ratio_upper_bound}]")

es = cma.CMAEvolutionStrategy(initial_normalized_params, sigma, opts)

# Optimization loop
while not es.stop():
    generation_counter += 1
    print(f"\nT1T2 Generation {generation_counter}")

    solutions = es.ask()
    fitness_values = []

    for i, solution in enumerate(solutions):
        solution = np.clip(solution, 0.0, 1.0)
        fitness = objective_function(solution)
        fitness_values.append(fitness)

    es.tell(solutions, fitness_values)

    current_best = es.best.f if es.best.f is not None else min(fitness_values)
    best_solution = es.best.x if es.best.x is not None else solutions[np.argmin(fitness_values)]
    best_solution = np.clip(best_solution, 0.0, 1.0)

    # Convert best solution to physical T1/T2 values and record to log file
    best_t1, best_t2 = transform_to_physical(best_solution)

    with open(generation_log, "a") as f:
        param_list = [str(generation_counter), f"{current_best:.6f}"]
        # Record T1 and T2 physical original values for all qubits
        for i in range(NUM_QUBITS):
            param_list.append(f"{best_t1[i]:.6e}")  # T1 value
            param_list.append(f"{best_t2[i]:.6e}")  # T2 value
        f.write(",".join(param_list) + "\n")

    print(f"T1T2 Generation {generation_counter}: Best = {current_best:.6f}")

    # Update shared parameter file every 10 generations
    if generation_counter % 10 == 0:
        t1t2_list = [{'t1': float(t1), 't2': float(t2)} for t1, t2 in zip(best_t1, best_t2)]
        success = update_t1t2_parameters(t1t2_list)
        if success:
            print(f"  >>> T1/T2 parameters updated to shared file (generation {generation_counter}) <<<")

# ==================== Save final results ====================
print("T1T2 optimizer: Starting to save final results...")

final_solution = np.clip(es.best.x, 0.0, 1.0)
final_fitness = es.best.f

try:
    final_t1, final_t2 = transform_to_physical(final_solution)
    final_t1t2_list = [{'t1': float(t1), 't2': float(t2)} for t1, t2 in zip(final_t1, final_t2)]

    success = update_t1t2_parameters(final_t1t2_list)

    print(f"\nT1T2 optimization completed!")
    print(f"Final objective value: {final_fitness:.6f}")
    print(f"Total generations: {generation_counter}")
    print(f"Total function evaluations: {function_evaluation_counter}")

    if success:
        print(f"Final T1/T2 parameters saved to shared file")
    else:
        print(f"Final T1/T2 parameter save failed")

    print(f"Output files:")
    print(f"- Shared JSON: {shared_json_file}")
    print(f"- Generations: {generation_log}")

except Exception as e:
    print(f"T1T2 optimizer: Error saving final results: {e}")

print("T1T2 optimizer: Program ended")
