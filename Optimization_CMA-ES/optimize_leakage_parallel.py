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

sigma = float(os.environ.get('CMA_LEAKAGE_SIGMA'))
popsize = int(os.environ.get('CMA_LEAKAGE_POPSIZE'))
maxiter = int(os.environ.get('CMA_LEAKAGE_MAXITER'))
tolfun = float(os.environ.get('CMA_LEAKAGE_TOLFUN'))
tolx = float(os.environ.get('CMA_LEAKAGE_TOLX'))

# Output log file
generation_log = os.environ.get('LEAKAGE_LOG_FILE')


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
        print(f"Leakage optimizer cannot read shared parameter file: {e}")
        raise e


def update_leakage_parameters(new_lp, new_sp):
    """Update LP/SP parameters in shared parameter file"""
    try:
        current_params = safe_read_json(shared_json_file)
        current_params['lp'] = new_lp
        current_params['sp'] = new_sp
        safe_write_json(shared_json_file, current_params)
        return True
    except Exception as e:
        print(f"Leakage optimizer failed to update shared parameters: {e}")
        return False


# ==================== Parameter normalization functions ====================
def normalize_params(physical_params, lower_bounds, upper_bounds):
    """Convert physical parameters to [0,1] normalized parameters"""
    physical_params = np.array(physical_params)
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)
    return (physical_params - lower_bounds) / (upper_bounds - lower_bounds)


def denormalize_params(normalized_params, lower_bounds, upper_bounds):
    """Convert [0,1] normalized parameters back to physical parameters"""
    normalized_params = np.array(normalized_params)
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)
    return normalized_params * (upper_bounds - lower_bounds) + lower_bounds


# ==================== Optimization-related functions ====================
function_evaluation_counter = 0

# Define physical parameter bounds
lp_lower_bound = 1e-5
lp_upper_bound = 5e-2
sp_lower_bound = 1e-3
sp_upper_bound = 5e-1


def calculate_objective(difference_matrix, num_mea, rounds_,
                        diagonal_weight=0.65, right_weight=0.35, second_right_weight=0.0):
    """Calculate composite objective function"""
    summed_differences = np.zeros(num_mea)

    for qubit_index in range(num_mea):
        start_index = qubit_index * rounds_
        end_index = (qubit_index + 1) * rounds_

        # Diagonal submatrix (excluding ±1 offset diagonals)
        submatrix = difference_matrix[start_index:end_index, start_index:end_index]
        one_offset = np.zeros_like(submatrix)

        diag1 = np.diag(submatrix, k=1)
        diag2 = np.diag(submatrix, k=-1)

        np.fill_diagonal(one_offset[1:], diag2)
        np.fill_diagonal(one_offset[:, 1:], diag1)

        dif_submatrix = submatrix - one_offset
        diagonal_contribution = np.sum(np.abs(dif_submatrix)) * diagonal_weight

        # Right neighbor submatrix (excluding only main diagonal)
        if qubit_index < num_mea - 1:
            right_start = (qubit_index + 1) * rounds_
            right_end = (qubit_index + 2) * rounds_
            right_neighbor_submatrix = difference_matrix[start_index:end_index, right_start:right_end]

            main_diagonal_matrix = np.zeros_like(right_neighbor_submatrix)
            main_diag = np.diag(right_neighbor_submatrix, k=0)
            main_diag1 = np.diag(right_neighbor_submatrix, k=1)
            main_diag2 = np.diag(right_neighbor_submatrix, k=-1)
            np.fill_diagonal(main_diagonal_matrix, main_diag)
            np.fill_diagonal(main_diagonal_matrix[1:], main_diag2)
            np.fill_diagonal(main_diagonal_matrix[:, 1:], main_diag1)

            right_dif_submatrix = right_neighbor_submatrix - main_diagonal_matrix
            right_contribution = np.sum(np.abs(right_dif_submatrix)) * right_weight
        else:
            right_contribution = 0

        # Second right neighbor submatrix (extract all)
        if qubit_index < num_mea - 2:
            second_right_start = (qubit_index + 2) * rounds_
            second_right_end = (qubit_index + 3) * rounds_
            second_right_submatrix = difference_matrix[start_index:end_index, second_right_start:second_right_end]
            second_right_contribution = np.sum(np.abs(second_right_submatrix)) * second_right_weight
        else:
            second_right_contribution = 0

        summed_differences[qubit_index] = diagonal_contribution + right_contribution + second_right_contribution

    return np.sum(summed_differences)

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
    """Objective function - get latest parameters for each evaluation"""
    global function_evaluation_counter
    function_evaluation_counter += 1

    # Get latest parameters
    current_params = get_latest_parameters()

    # Convert normalized parameters to physical space
    physical_lower_bounds = [lp_lower_bound] * NUM_QUBITS + [sp_lower_bound] * NUM_QUBITS
    physical_upper_bounds = [lp_upper_bound] * NUM_QUBITS + [sp_upper_bound] * NUM_QUBITS

    physical_params = denormalize_params(normalized_params, physical_lower_bounds, physical_upper_bounds)
    lp_params = physical_params[:NUM_QUBITS]
    sp_params = physical_params[NUM_QUBITS:]

    # Extract parameters that may have been updated by other optimizers
    spam_rates = current_params.get('spam_rates', [])
    sqg_fid = current_params.get('sqg_fid', [])
    ecr_fid = current_params.get('ecr_fid', [])
    t1_t2_values = [(d['t1'], d['t2']) for d in current_params.get('t1_t2_values', [])]
    ecr_lengths = current_params.get('ecr_lengths', [])
    rd_lengths = current_params.get('rd_length', [])
    sqg_length_list = current_params.get('sqg_length', [])
    sqg_lengths = sqg_length_list[0] if sqg_length_list else 0

    # Run simulation
    results_array = run_sampling(SHOTS_EXP, SHOTS, NUM_ROUNDS - 1, NUM_QUBITS,
                                 lp_params, sp_params, spam_rates, sqg_fid, ecr_fid,
                                 t1_t2_values, ecr_lengths, rd_lengths, sqg_lengths)

    M_sim = compute_correlation_matrix(results_array, NUM_QUBITS, NUM_ROUNDS, SHOTS * SHOTS_EXP)
    M_exp = correlation_matrix_training
    M_dif = M_sim - M_exp

    objective_value = calculate_objective(M_dif, num_mea=NUM_MEA, rounds_=NUM_ROUNDS)

    print(f"Leakage Eval {function_evaluation_counter:4d}: {objective_value:.6f}")
    print(f"  LP range: [{np.min(lp_params):.2e}, {np.max(lp_params):.2e}]")
    print(f"  SP range: [{np.min(sp_params):.2e}, {np.max(sp_params):.2e}]")
    sys.stdout.flush()

    return objective_value


# ==================== Initialize shared parameter file ====================
initial_json_file = os.environ.get('INITIAL_PARAMS_FILE')

if not os.path.exists(shared_json_file):
    print("Leakage optimizer: Shared parameter file does not exist, creating...")
    try:
        with open(initial_json_file, 'r') as f:
            initial_params = json.load(f)
        safe_write_json(shared_json_file, initial_params)
        print(f"Leakage optimizer: Shared parameter file created: {shared_json_file}")
    except Exception as e:
        print(f"Leakage optimizer: Failed to create shared parameter file: {e}")
        sys.exit(1)

# Get initial parameters and prepare normalization
initial_params = get_latest_parameters()
initial_lp = np.array(initial_params.get('lp', []))
initial_sp = np.array(initial_params.get('sp', []))

print(f"Leakage optimizer started")
print(f"Initial LP: [{np.min(initial_lp):.2e}, {np.max(initial_lp):.2e}]")
print(f"Initial SP: [{np.min(initial_sp):.2e}, {np.max(initial_sp):.2e}]")

# Ensure parameters are within bounds
initial_lp = np.clip(initial_lp, lp_lower_bound, lp_upper_bound)
initial_sp = np.clip(initial_sp, sp_lower_bound, sp_upper_bound)

# Convert to normalized space
physical_lower_bounds = [lp_lower_bound] * NUM_QUBITS + [sp_lower_bound] * NUM_QUBITS
physical_upper_bounds = [lp_upper_bound] * NUM_QUBITS + [sp_upper_bound] * NUM_QUBITS
initial_physical_params = np.concatenate([initial_lp, initial_sp])
initial_normalized_params = normalize_params(initial_physical_params, physical_lower_bounds, physical_upper_bounds)

# ==================== Optimization setup and execution ====================
normalized_lower_bounds = [0.0] * (2 * NUM_QUBITS)
normalized_upper_bounds = [1.0] * (2 * NUM_QUBITS)

# Create generations file
with open(generation_log, "w") as f:
    f.write("Generation,BestObjective,BestParameters\n")

# Use dynamic configuration
generation_counter = 0


opts = {
    'bounds': [normalized_lower_bounds, normalized_upper_bounds],
    'maxiter': maxiter,
    'popsize': popsize,
    'verb_log': 0,
    'verb_disp': 0,
    'tolfun': tolfun,
    'tolx': tolx,
    'seed': 42,
}

print("Leakage optimizer: Starting CMA-ES optimization...")
print(f"Population size: {popsize}")
print(f"Max generations: {opts['maxiter']}")
print(f"Parameter dimensions: {2 * NUM_QUBITS} (LP + SP)")

es = cma.CMAEvolutionStrategy(initial_normalized_params, sigma, opts)

# Optimization loop
while not es.stop():
    generation_counter += 1
    print(f"\nLeakage Generation {generation_counter}")

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

    print(f"Leakage Generation {generation_counter}: Best = {current_best:.6f}")

    # Update shared parameter file every 10 generations
    if generation_counter % 10 == 0:
        # Convert best solution back to physical space
        best_physical = denormalize_params(best_solution, physical_lower_bounds, physical_upper_bounds)
        best_lp = best_physical[:NUM_QUBITS].tolist()
        best_sp = best_physical[NUM_QUBITS:].tolist()

        success = update_leakage_parameters(best_lp, best_sp)
        if success:
            print(f"  >>> LP/SP parameters updated to shared file (generation {generation_counter}) <<<")
        else:
            print(f"  >>> LP/SP parameter update failed (generation {generation_counter}) <<<")

# ==================== Save final results ====================
final_solution = es.best.x
final_fitness = es.best.f

# Convert final solution back to physical space
final_physical = denormalize_params(final_solution, physical_lower_bounds, physical_upper_bounds)
final_lp = final_physical[:NUM_QUBITS].tolist()
final_sp = final_physical[NUM_QUBITS:].tolist()

# Update final parameters to shared file
success = update_leakage_parameters(final_lp, final_sp)

print(f"\nLeakage optimization completed!")
print(f"Final objective value: {final_fitness:.6f}")
print(f"Total generations: {generation_counter}")
print(f"Total function evaluations: {function_evaluation_counter}")
print(f"Final LP range: [{np.min(final_lp):.2e}, {np.max(final_lp):.2e}]")
print(f"Final SP range: [{np.min(final_sp):.2e}, {np.max(final_sp):.2e}]")

if success:
    print(f"Final LP/SP parameters saved to shared file: {shared_json_file}")
else:
    print(f"Final LP/SP parameter save failed")

print(f"Output files:")
print(f"- Shared JSON: {shared_json_file}")
print(f"- Generations: {generation_log}")