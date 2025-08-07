import numpy as np
import sys
import os
import json
from typing import Dict, List, Tuple

# Set absolute paths - please modify the following paths according to actual situation
BASE_DIR = r"E:\Quantum-Noise-Model-Simulation\Quantum_Noise_Model_Simulation_main"
PROJECT_ROOT = BASE_DIR
PARAM_FILES_DIR = r"E:\Quamtum-Noise-Model-Simulation\Quantum_Noise_Model_Simulation_main\Verification"  # Parameter files directory
RESULTS_BASE_DIR = r"E:\Quamtum-Noise-Model-Simulation\Quantum_Noise_Model_Simulation_main\Verification"  # Experimental results root directory

# Add project path to system path
sys.path.append(PROJECT_ROOT)

import sim.sim_Repetition_X
import sim.sim_Repetition_Z

# Chip configuration - using absolute paths
CHIP_CONFIGS = {
    'ibm_brisbane': {
        'param_file': os.path.join(PARAM_FILES_DIR, 'optimized_parameters_ibm_brisbane.json'),
        'exp_data_X_dir': os.path.join(RESULTS_BASE_DIR, 'results_single_repx_ibm_brisbane', 'results_extract_RepX',
                                       'averaged'),
        'exp_data_Z_dir': os.path.join(RESULTS_BASE_DIR, 'results_single_repz_ibm_brisbane', 'results_extract_RepZ',
                                       'averaged')
    },
    'ibm_sherbrooke': {
        'param_file': os.path.join(PARAM_FILES_DIR, 'optimized_parameters_ibm_sherbrooke.json'),
        'exp_data_X_dir': os.path.join(RESULTS_BASE_DIR, 'results_single_repx_ibm_sherbrooke', 'results_extract_RepX',
                                       'averaged'),
        'exp_data_Z_dir': os.path.join(RESULTS_BASE_DIR, 'results_single_repz_ibm_sherbrooke', 'results_extract_RepZ',
                                       'averaged')
    },
    'ibm_torino': {
        'param_file': os.path.join(PARAM_FILES_DIR, 'optimized_parameters_ibm_torino.json'),
        'exp_data_X_dir': os.path.join(RESULTS_BASE_DIR, 'results_single_repx_ibm_torino', 'results_extract_RepX',
                                       'averaged'),
        'exp_data_Z_dir': os.path.join(RESULTS_BASE_DIR, 'results_single_repz_ibm_torino', 'results_extract_RepZ',
                                       'averaged')
    },
    'tencent': {
        'param_file': os.path.join(PARAM_FILES_DIR, 'optimized_parameters_tencent.json'),
        'exp_data_X_dir': os.path.join(RESULTS_BASE_DIR, 'results_single_repx_tencent', 'results_extract_RepX',
                                       'averaged'),
        'exp_data_Z_dir': os.path.join(RESULTS_BASE_DIR, 'results_single_repz_tencent', 'results_extract_RepZ',
                                       'averaged')
    },
    'tianyan': {
        'param_file': os.path.join(PARAM_FILES_DIR, 'optimized_parameters_tianyan.json'),
        'exp_data_X_dir': os.path.join(RESULTS_BASE_DIR, 'results_single_repx_tianyan', 'results_extract_RepX',
                                       'averaged'),
        'exp_data_Z_dir': os.path.join(RESULTS_BASE_DIR, 'results_single_repz_tianyan', 'results_extract_RepZ',
                                       'averaged')
    }
}

# Traditional model configuration
TRADITIONAL_MODELS = {
    'circuit': 'circuit',
    'codecapacity': 'codecapacity',
    'phenomenological': 'phenomenological',
    'SD6': 'SD6',
    'SI1000': 'SI1000'
}

# Real chip configuration
REAL_CHIPS = {
    'ibm_brisbane': 'ibm_brisbane',
    'ibm_sherbrooke': 'ibm_sherbrooke',
    'ibm_torino': 'ibm_torino',
    'tencent': 'tencent',
    'tianyan': 'tianyan'
}

# P-values file path
P_VALUES_FILE = os.path.join(RESULTS_BASE_DIR, 'optimal_p_values.json')

# Traditional model experimental data configuration
TRADITIONAL_MODEL_CONFIGS = {
    'circuit': {
        'exp_data_X_dir': os.path.join(RESULTS_BASE_DIR, 'results_single_repx_circuit'),
        'exp_data_Z_dir': os.path.join(RESULTS_BASE_DIR, 'results_single_repz_circuit')
    },
    'codecapacity': {
        'exp_data_X_dir': os.path.join(RESULTS_BASE_DIR, 'results_single_repx_codecapacity'),
        'exp_data_Z_dir': os.path.join(RESULTS_BASE_DIR, 'results_single_repz_codecapacity')
    },
    'phenomenological': {
        'exp_data_X_dir': os.path.join(RESULTS_BASE_DIR, 'results_single_repx_phenomenological'),
        'exp_data_Z_dir': os.path.join(RESULTS_BASE_DIR, 'results_single_repz_phenomenological')
    },
    'SD6': {
        'exp_data_X_dir': os.path.join(RESULTS_BASE_DIR, 'results_single_repx_SD6'),
        'exp_data_Z_dir': os.path.join(RESULTS_BASE_DIR, 'results_single_repz_SD6')
    },
    'SI1000': {
        'exp_data_X_dir': os.path.join(RESULTS_BASE_DIR, 'results_single_repx_SI1000'),
        'exp_data_Z_dir': os.path.join(RESULTS_BASE_DIR, 'results_single_repz_SI1000')
    }
}

# Real chip experimental data configuration
REAL_CHIP_CONFIGS = {
    'ibm_brisbane': {
        'exp_data_X_dir': os.path.join(RESULTS_BASE_DIR, 'results_single_repx_ibm_brisbane', 'results_extract_RepX',
                                       'averaged'),
        'exp_data_Z_dir': os.path.join(RESULTS_BASE_DIR, 'results_single_repz_ibm_brisbane', 'results_extract_RepZ',
                                       'averaged')
    },
    'ibm_sherbrooke': {
        'exp_data_X_dir': os.path.join(RESULTS_BASE_DIR, 'results_single_repx_ibm_sherbrooke', 'results_extract_RepX',
                                       'averaged'),
        'exp_data_Z_dir': os.path.join(RESULTS_BASE_DIR, 'results_single_repz_ibm_sherbrooke', 'results_extract_RepZ',
                                       'averaged')
    },
    'ibm_torino': {
        'exp_data_X_dir': os.path.join(RESULTS_BASE_DIR, 'results_single_repx_ibm_torino', 'results_extract_RepX',
                                       'averaged'),
        'exp_data_Z_dir': os.path.join(RESULTS_BASE_DIR, 'results_single_repz_ibm_torino', 'results_extract_RepZ',
                                       'averaged')
    },
    'tencent': {
        'exp_data_X_dir': os.path.join(RESULTS_BASE_DIR, 'results_single_repx_tencent', 'results_extract_RepX',
                                       'averaged'),
        'exp_data_Z_dir': os.path.join(RESULTS_BASE_DIR, 'results_single_repz_tencent', 'results_extract_RepZ',
                                       'averaged')
    },
    'tianyan': {
        'exp_data_X_dir': os.path.join(RESULTS_BASE_DIR, 'results_single_repx_tianyan', 'results_extract_RepX',
                                       'averaged'),
        'exp_data_Z_dir': os.path.join(RESULTS_BASE_DIR, 'results_single_repz_tianyan', 'results_extract_RepZ',
                                       'averaged')
    }
}

TIMES = 10
SHOTS_EXP = 1
SHOTS = 4096 * TIMES

# Parameter configuration
NUM_QUBITS_LIST = [5, 9, 13, 17, 21]
def load_chip_parameters(param_file_path: str) -> Dict:
    """Load chip parameters"""
    with open(param_file_path, 'r') as json_file:
        parameters = json.load(json_file)

    return {
        'spam_rates': parameters.get('spam_rates', []),
        'spam_rates_initial': parameters.get('spam_rates_initial', []),
        'lp': parameters.get('lp', []),
        'sp': parameters.get('sp', []),
        'ecr_fid': parameters.get('ecr_fid', []),
        'sqg_fid': parameters.get('sqg_fid', []),
        't1_t2_values': [(d['t1'], d['t2']) for d in parameters.get('t1_t2_values', [])],
        'ecr_lengths': parameters.get('ecr_lengths', []),
        'rd_lengths': parameters.get('rd_length', [])
    }


def load_experimental_data(file_path: str) -> Dict:
    """Load experimental data"""
    with open(file_path, 'r') as f:
        return json.load(f)


def load_p_values(p_values_file_path: str) -> Dict:
    """Load P-values data"""
    with open(p_values_file_path, 'r') as f:
        return json.load(f)


def preload_chip_experimental_data(chip_name: str, config: Dict) -> Tuple[Dict, Dict]:
    """Preload experimental data for specified chip"""
    print(f"📊 Preloading experimental data for {chip_name}...")

    exp_data_X = {}
    exp_data_Z = {}

    for num_qubits in NUM_QUBITS_LIST:
        try:
            # X basis data
            x_file_path = os.path.join(config['exp_data_X_dir'], f'avg_counts_n{num_qubits}.json')
            exp_data_X[num_qubits] = load_experimental_data(x_file_path)

            # Z basis data
            z_file_path = os.path.join(config['exp_data_Z_dir'], f'avg_counts_n{num_qubits}.json')
            exp_data_Z[num_qubits] = load_experimental_data(z_file_path)

            print(
                f"   ✓ n={num_qubits}: X basis ({len(exp_data_X[num_qubits])} states), Z basis ({len(exp_data_Z[num_qubits])} states)")

        except FileNotFoundError as e:
            print(f"   ❌ n={num_qubits}: File not found - {e}")
            raise

    return exp_data_X, exp_data_Z


def load_traditional_model_data_by_p_value(model_name: str, config: Dict, p_value: float) -> Tuple[Dict, Dict]:
    """Load traditional model experimental data based on P-value"""
    print(f"📊 Loading experimental data for {model_name} model (P={p_value:.6f})...")

    exp_data_X = {}
    exp_data_Z = {}

    for num_qubits in NUM_QUBITS_LIST:
        try:
            # X basis data - find corresponding P-value file
            x_subdir = os.path.join(config['exp_data_X_dir'], f'n{num_qubits}')
            x_target_file = os.path.join(x_subdir, f'p_{p_value:.6f}.json')

            # If exact match doesn't exist, find closest P-value file
            if not os.path.exists(x_target_file):
                x_target_file = find_closest_p_value_file(x_subdir, p_value)

            if x_target_file and os.path.exists(x_target_file):
                with open(x_target_file, 'r') as f:
                    exp_data_X[num_qubits] = json.load(f)
                print(f"   ✓ X basis n={num_qubits}: Loaded file {os.path.basename(x_target_file)}")
            else:
                exp_data_X[num_qubits] = {}
                print(f"   ❌ X basis n={num_qubits}: P-value file not found")

            # Z basis data - similar processing
            z_subdir = os.path.join(config['exp_data_Z_dir'], f'n{num_qubits}')
            z_target_file = os.path.join(z_subdir, f'p_{p_value:.6f}.json')

            if not os.path.exists(z_target_file):
                z_target_file = find_closest_p_value_file(z_subdir, p_value)

            if z_target_file and os.path.exists(z_target_file):
                with open(z_target_file, 'r') as f:
                    exp_data_Z[num_qubits] = json.load(f)
                print(f"   ✓ Z basis n={num_qubits}: Loaded file {os.path.basename(z_target_file)}")
            else:
                exp_data_Z[num_qubits] = {}
                print(f"   ❌ Z basis n={num_qubits}: P-value file not found")

        except Exception as e:
            print(f"   ❌ n={num_qubits}: Data loading error - {e}")
            exp_data_X[num_qubits] = {}
            exp_data_Z[num_qubits] = {}

    return exp_data_X, exp_data_Z


def find_closest_p_value_file(directory: str, target_p_value: float) -> str:
    """Find the file with P-value closest to target in directory"""
    if not os.path.exists(directory):
        return None

    best_file = None
    min_diff = float('inf')

    for filename in os.listdir(directory):
        if filename.startswith('p_') and filename.endswith('.json'):
            try:
                # Extract P-value from filename
                p_str = filename[2:-5]  # Remove 'p_' and '.json'
                file_p_value = float(p_str)

                diff = abs(file_p_value - target_p_value)
                if diff < min_diff:
                    min_diff = diff
                    best_file = os.path.join(directory, filename)

            except ValueError:
                continue  # Skip unparseable filenames

    return best_file


def load_real_chip_experimental_data(chip_name: str, config: Dict) -> Tuple[Dict, Dict]:
    """Load real chip experimental data"""
    print(f"📊 Loading experimental data for {chip_name} chip...")

    exp_data_X = {}
    exp_data_Z = {}

    for num_qubits in NUM_QUBITS_LIST:
        try:
            # X basis data
            x_file_path = os.path.join(config['exp_data_X_dir'], f'avg_counts_n{num_qubits}.json')
            with open(x_file_path, 'r') as f:
                exp_data_X[num_qubits] = json.load(f)

            # Z basis data
            z_file_path = os.path.join(config['exp_data_Z_dir'], f'avg_counts_n{num_qubits}.json')
            with open(z_file_path, 'r') as f:
                exp_data_Z[num_qubits] = json.load(f)

            print(
                f"   ✓ n={num_qubits}: X basis ({len(exp_data_X[num_qubits])} states), Z basis ({len(exp_data_Z[num_qubits])} states)")

        except FileNotFoundError as e:
            print(f"   ❌ n={num_qubits}: File not found - {e}")
            exp_data_X[num_qubits] = {}
            exp_data_Z[num_qubits] = {}

    return exp_data_X, exp_data_Z

def analyze_results_for_l1(results_array) -> Dict[str, int]:
    """Version optimized specifically for L1 distance calculation"""
    counts = {}
    for row in results_array:
        bitstring = ''.join(str(int(b)) for b in row)
        reversed_bitstring = bitstring[::-1]
        counts[reversed_bitstring] = counts.get(reversed_bitstring, 0) + 1
    return counts


def calculate_l1_distance(sim_counts: Dict[str, int], exp_counts: Dict[str, int],
                          total_shots_sim: int, total_shots_exp: int) -> float:
    """Calculate L1 distance between two distributions"""
    # Get all states that appear
    all_states = set(sim_counts.keys()) | set(exp_counts.keys())

    l1_distance = 0.0
    for state in all_states:
        # Convert to probabilities
        p_sim = sim_counts.get(state, 0) / total_shots_sim
        p_exp = exp_counts.get(state, 0) / total_shots_exp
        l1_distance += abs(p_sim - p_exp)

    return l1_distance


def calculate_l1_distance_simple(data1: Dict[str, int], data2: Dict[str, int]) -> float:
    """Calculate L1 distance between two distributions (simplified version)"""
    total_shots_1 = sum(data1.values()) if data1 else 0
    total_shots_2 = sum(data2.values()) if data2 else 0

    if total_shots_1 == 0 or total_shots_2 == 0:
        return float('inf')  # Return infinity if no data

    # Get all states that appear
    all_states = set(data1.keys()) | set(data2.keys())

    l1_distance = 0.0
    for state in all_states:
        # Convert to probabilities
        p1 = data1.get(state, 0) / total_shots_1
        p2 = data2.get(state, 0) / total_shots_2
        l1_distance += abs(p1 - p2)

    return l1_distance


def calculate_chip_l1_distances(chip_name: str, config: Dict, parameters: Dict) -> Dict:
    """Calculate L1 distances for a single chip"""
    print(f"\n{'=' * 80}")
    print(f"Calculating L1 distances for {chip_name.upper()} chip")
    print(f"{'=' * 80}")

    # Preload experimental data
    exp_data_X, exp_data_Z = preload_chip_experimental_data(chip_name, config)

    results = {
        'chip_name': chip_name,
        'qubit_results_X': {},
        'qubit_results_Z': {},
        'total_l1_X': 0,
        'total_l1_Z': 0,
        'total_l1': 0
    }

    print(f"\nStarting L1 distance calculation for different qubit counts:")

    for num_qubits in NUM_QUBITS_LIST:
        print(f"\n--- Calculating n={num_qubits} qubits ---")

        # X basis simulation
        print(f"  Running X basis simulation...")
        sim_result_array_X = sim.sim_Repetition_X.run_sampling(
            SHOTS_EXP, SHOTS, 0, num_qubits,
            parameters['lp'], parameters['sp'], parameters['spam_rates'],
            parameters['spam_rates_initial'], parameters['sqg_fid'],
            parameters['ecr_fid'], parameters['t1_t2_values'],
            parameters['ecr_lengths'], parameters['rd_lengths']
        )
        sim_counts_X = analyze_results_for_l1(sim_result_array_X)

        # Calculate X basis L1 distance
        total_shots_sim_X = sum(sim_counts_X.values())
        total_shots_exp_X = sum(exp_data_X[num_qubits].values())
        l1_dist_X = calculate_l1_distance(sim_counts_X, exp_data_X[num_qubits],
                                          total_shots_sim_X, total_shots_exp_X)

        results['qubit_results_X'][num_qubits] = l1_dist_X
        results['total_l1_X'] += l1_dist_X

        # Z basis simulation
        print(f"  Running Z basis simulation...")
        sim_result_array_Z = sim.sim_Repetition_Z.run_sampling(
            SHOTS_EXP, SHOTS, 0, num_qubits,
            parameters['lp'], parameters['sp'], parameters['spam_rates'],
            parameters['spam_rates_initial'], parameters['sqg_fid'],
            parameters['ecr_fid'], parameters['t1_t2_values'],
            parameters['ecr_lengths'], parameters['rd_lengths']
        )
        sim_counts_Z = analyze_results_for_l1(sim_result_array_Z)

        # Calculate Z basis L1 distance
        total_shots_sim_Z = sum(sim_counts_Z.values())
        total_shots_exp_Z = sum(exp_data_Z[num_qubits].values())
        l1_dist_Z = calculate_l1_distance(sim_counts_Z, exp_data_Z[num_qubits],
                                          total_shots_sim_Z, total_shots_exp_Z)

        results['qubit_results_Z'][num_qubits] = l1_dist_Z
        results['total_l1_Z'] += l1_dist_Z

        print(f"  Results: L1_X = {l1_dist_X:.6f}, L1_Z = {l1_dist_Z:.6f}")

    # Calculate total L1 distance
    results['total_l1'] = results['total_l1_X'] + results['total_l1_Z']

    return results


def calculate_comparison_l1_distances(traditional_model: str, real_chip: str, p_value: float) -> Dict:
    """Calculate L1 distance comparison between traditional model and real chip"""
    print(f"\n{'=' * 120}")
    print(f"Comparing {traditional_model.upper()} model (optimized P={p_value:.6f} for {real_chip}) vs {real_chip.upper()} chip")
    print(f"{'=' * 120}")

    # Load traditional model experimental data
    traditional_config = TRADITIONAL_MODEL_CONFIGS[traditional_model]
    traditional_exp_data_X, traditional_exp_data_Z = load_traditional_model_data_by_p_value(
        traditional_model, traditional_config, p_value
    )

    # Load real chip experimental data
    chip_config = REAL_CHIP_CONFIGS[real_chip]
    chip_exp_data_X, chip_exp_data_Z = load_real_chip_experimental_data(
        real_chip, chip_config
    )

    results = {
        'traditional_model': traditional_model,
        'real_chip': real_chip,
        'p_value': p_value,
        'qubit_results_X': {},
        'qubit_results_Z': {},
        'total_l1_X': 0,
        'total_l1_Z': 0,
        'total_l1': 0
    }

    print(f"\nStarting L1 distance calculation for different qubit counts:")

    for num_qubits in NUM_QUBITS_LIST:
        print(f"\n--- Calculating n={num_qubits} qubits ---")

        # Calculate X basis L1 distance
        l1_dist_X = calculate_l1_distance_simple(
            traditional_exp_data_X[num_qubits],
            chip_exp_data_X[num_qubits]
        )
        results['qubit_results_X'][num_qubits] = l1_dist_X
        if l1_dist_X != float('inf'):
            results['total_l1_X'] += l1_dist_X

        # Calculate Z basis L1 distance
        l1_dist_Z = calculate_l1_distance_simple(
            traditional_exp_data_Z[num_qubits],
            chip_exp_data_Z[num_qubits]
        )
        results['qubit_results_Z'][num_qubits] = l1_dist_Z
        if l1_dist_Z != float('inf'):
            results['total_l1_Z'] += l1_dist_Z

        print(f"  Results: L1_X = {l1_dist_X:.6f}, L1_Z = {l1_dist_Z:.6f}")

    # Calculate total L1 distance
    results['total_l1'] = results['total_l1_X'] + results['total_l1_Z']

    return results

def print_chip_results(results: Dict):
    """Print results for a single chip"""
    chip_name = results['chip_name']

    print(f"\n{'=' * 80}")
    print(f"Detailed L1 distance results for {chip_name.upper()} chip")
    print(f"{'=' * 80}")

    # X basis results
    print(f"\nX basis measurement results:")
    print(f"{'Qubits':<10} {'L1 Distance':<12}")
    print(f"{'-' * 22}")
    for num_qubits in NUM_QUBITS_LIST:
        l1_x = results['qubit_results_X'][num_qubits]
        print(f"{num_qubits:<10} {l1_x:<12.6f}")
    print(f"{'-' * 22}")
    print(f"{'X Total':<10} {results['total_l1_X']:<12.6f}")

    # Z basis results
    print(f"\nZ basis measurement results:")
    print(f"{'Qubits':<10} {'L1 Distance':<12}")
    print(f"{'-' * 22}")
    for num_qubits in NUM_QUBITS_LIST:
        l1_z = results['qubit_results_Z'][num_qubits]
        print(f"{num_qubits:<10} {l1_z:<12.6f}")
    print(f"{'-' * 22}")
    print(f"{'Z Total':<10} {results['total_l1_Z']:<12.6f}")

    # Summary
    print(f"\nSummary:")
    print(f"Total L1 distance: {results['total_l1']:.6f}")
    print(f"X basis proportion: {results['total_l1_X'] / results['total_l1'] * 100:.2f}%")
    print(f"Z basis proportion: {results['total_l1_Z'] / results['total_l1'] * 100:.2f}%")


def print_comparison_results(results: Dict):
    """Print comparison results"""
    traditional_model = results['traditional_model']
    real_chip = results['real_chip']
    p_value = results['p_value']

    print(f"\n{'=' * 120}")
    print(f"Detailed L1 distance results for {traditional_model.upper()} model (P={p_value:.6f}) vs {real_chip.upper()} chip")
    print(f"{'=' * 120}")

    # X basis results
    print(f"\nX basis measurement results:")
    print(f"{'Qubits':<10} {'L1 Distance':<15}")
    print(f"{'-' * 25}")
    for num_qubits in NUM_QUBITS_LIST:
        l1_x = results['qubit_results_X'][num_qubits]
        if l1_x == float('inf'):
            print(f"{num_qubits:<10} {'∞ (no data)':<15}")
        else:
            print(f"{num_qubits:<10} {l1_x:<15.6f}")
    print(f"{'-' * 25}")
    print(f"{'X Total':<10} {results['total_l1_X']:<15.6f}")

    # Z basis results
    print(f"\nZ basis measurement results:")
    print(f"{'Qubits':<10} {'L1 Distance':<15}")
    print(f"{'-' * 25}")
    for num_qubits in NUM_QUBITS_LIST:
        l1_z = results['qubit_results_Z'][num_qubits]
        if l1_z == float('inf'):
            print(f"{num_qubits:<10} {'∞ (no data)':<15}")
        else:
            print(f"{num_qubits:<10} {l1_z:<15.6f}")
    print(f"{'-' * 25}")
    print(f"{'Z Total':<10} {results['total_l1_Z']:<15.6f}")

    # Summary
    print(f"\nSummary:")
    print(f"Total L1 distance: {results['total_l1']:.6f}")
    if results['total_l1'] > 0:
        print(f"X basis proportion: {results['total_l1_X'] / results['total_l1'] * 100:.2f}%")
        print(f"Z basis proportion: {results['total_l1_Z'] / results['total_l1'] * 100:.2f}%")


def print_summary_results(summary_results: List[Dict]):
    """Print summary results for all chips"""
    print(f"\n{'=' * 100}")
    print(f"Summary of L1 distances for all chips")
    print(f"{'=' * 100}")

    print(f"{'Chip Name':<20} {'Total L1':<12} {'X Basis L1':<12} {'Z Basis L1':<12} {'X Ratio':<10} {'Z Ratio':<10}")
    print(f"{'-' * 86}")

    for result in summary_results:
        chip_name = result['chip_name']
        total_l1 = result['total_l1']
        l1_x = result['total_l1_X']
        l1_z = result['total_l1_Z']
        x_ratio = l1_x / total_l1 * 100 if total_l1 > 0 else 0
        z_ratio = l1_z / total_l1 * 100 if total_l1 > 0 else 0

        print(f"{chip_name:<20} {total_l1:<12.6f} {l1_x:<12.6f} {l1_z:<12.6f} {x_ratio:<10.2f}% {z_ratio:<10.2f}%")

    # Find best chip
    if summary_results:
        best_chip = min(summary_results, key=lambda x: x['total_l1'])
        worst_chip = max(summary_results, key=lambda x: x['total_l1'])

        print(f"\n🏆 Best matching chip: {best_chip['chip_name']} (L1 distance: {best_chip['total_l1']:.6f})")
        print(f"🔻 Worst matching chip: {worst_chip['chip_name']} (L1 distance: {worst_chip['total_l1']:.6f})")

        avg_l1 = sum(r['total_l1'] for r in summary_results) / len(summary_results)
        print(f"📊 Average L1 distance: {avg_l1:.6f}")


def print_comparison_summary(summary_results: List[Dict]):
    """Print comparison summary results"""
    print(f"\n{'=' * 150}")
    print(f"Traditional Model vs Real Chip L1 Distance Summary")
    print(f"{'=' * 150}")

    print(
        f"{'Traditional Model':<15} {'Real Chip':<15} {'P Value':<12} {'Total L1':<15} {'X Basis L1':<15} {'Z Basis L1':<15} {'X Ratio':<10} {'Z Ratio':<10}")
    print(f"{'-' * 140}")

    for result in summary_results:
        traditional_model = result['traditional_model']
        real_chip = result['real_chip']
        p_value = result['p_value']
        total_l1 = result['total_l1']
        l1_x = result['total_l1_X']
        l1_z = result['total_l1_Z']
        x_ratio = l1_x / total_l1 * 100 if total_l1 > 0 else 0
        z_ratio = l1_z / total_l1 * 100 if total_l1 > 0 else 0

        print(
            f"{traditional_model:<15} {real_chip:<15} {p_value:<12.6f} {total_l1:<15.6f} {l1_x:<15.6f} {l1_z:<15.6f} {x_ratio:<10.2f}% {z_ratio:<10.2f}%")

    # Analyze best matches
    if summary_results:
        print(f"\n🏆 Analysis results:")

        # Find best matching chip for each traditional model
        model_best_matches = {}
        for result in summary_results:
            model = result['traditional_model']
            if model not in model_best_matches or result['total_l1'] < model_best_matches[model]['total_l1']:
                model_best_matches[model] = result

        print(f"\nBest matching chip for each traditional model:")
        for model, best_match in model_best_matches.items():
            print(f"  {model:<15} -> {best_match['real_chip']:<15} (L1 distance: {best_match['total_l1']:.6f})")

        # Find best matching traditional model for each chip
        chip_best_matches = {}
        for result in summary_results:
            chip = result['real_chip']
            if chip not in chip_best_matches or result['total_l1'] < chip_best_matches[chip]['total_l1']:
                chip_best_matches[chip] = result

        print(f"\nBest matching traditional model for each real chip:")
        for chip, best_match in chip_best_matches.items():
            print(f"  {chip:<15} -> {best_match['traditional_model']:<15} (L1 distance: {best_match['total_l1']:.6f})")


def calculate_all_chips_l1_distances():
    """Calculate L1 distances for all chips"""
    print(f"\n{'=' * 100}")
    print(f"Starting L1 distance calculation for all chips")
    print(f"{'=' * 100}")

    all_results = {}
    summary_results = []

    for chip_name, config in CHIP_CONFIGS.items():
        try:
            # Load parameters
            param_file_path = config['param_file']  # Already absolute path
            parameters = load_chip_parameters(param_file_path)

            print(f"\n✅ {chip_name} parameters loaded successfully")
            print(f"   spam_rates: {len(parameters['spam_rates'])} parameters")
            print(f"   ecr_fid: {len(parameters['ecr_fid'])} parameters")
            print(f"   t1_t2_values: {len(parameters['t1_t2_values'])} parameters")

            # Calculate L1 distances
            chip_results = calculate_chip_l1_distances(chip_name, config, parameters)
            all_results[chip_name] = chip_results

            # Add to summary
            summary_results.append({
                'chip_name': chip_name,
                'total_l1': chip_results['total_l1'],
                'total_l1_X': chip_results['total_l1_X'],
                'total_l1_Z': chip_results['total_l1_Z']
            })

            # Print results
            print_chip_results(chip_results)

        except Exception as e:
            print(f"❌ Error processing {chip_name}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary results for all chips
    print_summary_results(summary_results)

    return all_results


def calculate_all_traditional_vs_chip_comparisons():
    """Calculate comparisons between all traditional models and real chips"""
    print(f"\n{'=' * 120}")
    print(f"Starting L1 distance comparison calculation for all traditional models vs real chips")
    print(f"{'=' * 120}")

    # Load P-values data
    try:
        p_values_data = load_p_values(P_VALUES_FILE)
        print(f"✅ P-values data loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load P-values data: {e}")
        return {}

    all_comparison_results = {}
    summary_results = []

    # Iterate through all combinations of chips and traditional models
    for real_chip in REAL_CHIPS.keys():
        if real_chip not in p_values_data:
            print(f"⚠️  Skipping chip {real_chip}: P-values data not found")
            continue

        chip_p_values = p_values_data[real_chip]
        print(f"\n🔧 Processing real chip: {real_chip}")

        for traditional_model in TRADITIONAL_MODELS.keys():
            if traditional_model not in chip_p_values:
                print(f"⚠️  Skipping {real_chip}-{traditional_model}: P-values data not found")
                continue

            # Get the specific P-value for this chip corresponding to this traditional model
            model_p_value = chip_p_values[traditional_model]['optimal_p']

            print(f"\n🔍 Starting comparison: {traditional_model} (P={model_p_value:.6f}) vs {real_chip}")

            try:
                # Calculate L1 distance
                comparison_results = calculate_comparison_l1_distances(
                    traditional_model, real_chip, model_p_value
                )

                comparison_key = f"{traditional_model}_vs_{real_chip}"
                all_comparison_results[comparison_key] = comparison_results

                # Add to summary
                summary_results.append({
                    'traditional_model': traditional_model,
                    'real_chip': real_chip,
                    'p_value': model_p_value,
                    'total_l1': comparison_results['total_l1'],
                    'total_l1_X': comparison_results['total_l1_X'],
                    'total_l1_Z': comparison_results['total_l1_Z']
                })

                # Print results
                print_comparison_results(comparison_results)

            except Exception as e:
                print(f"❌ Error comparing {traditional_model} vs {real_chip}: {e}")
                import traceback
                traceback.print_exc()

    # Print summary results
    print_comparison_summary(summary_results)

    return all_comparison_results

def save_comprehensive_results(all_chip_results: Dict, all_comparison_results: Dict):
    """Save comprehensive results to JSON file"""
    comprehensive_results = {}

    # Process each chip
    for chip_name in REAL_CHIPS.keys():
        comprehensive_results[chip_name] = {}

        # Add simulation model results (if exists)
        if chip_name in all_chip_results:
            sim_result = all_chip_results[chip_name]
            comprehensive_results[chip_name]['simulation_model'] = {
                'X_basis': sim_result['qubit_results_X'],
                'Z_basis': sim_result['qubit_results_Z'],
                'total_l1_X': sim_result['total_l1_X'],
                'total_l1_Z': sim_result['total_l1_Z'],
                'total_l1': sim_result['total_l1']
            }

        # Add traditional model results
        for traditional_model in TRADITIONAL_MODELS.keys():
            comparison_key = f"{traditional_model}_vs_{chip_name}"
            if comparison_key in all_comparison_results:
                comp_result = all_comparison_results[comparison_key]
                comprehensive_results[chip_name][traditional_model] = {
                    'p_value': comp_result['p_value'],
                    'X_basis': comp_result['qubit_results_X'],
                    'Z_basis': comp_result['qubit_results_Z'],
                    'total_l1_X': comp_result['total_l1_X'],
                    'total_l1_Z': comp_result['total_l1_Z'],
                    'total_l1': comp_result['total_l1']
                }

    # Save to JSON file
    output_file = os.path.join(RESULTS_BASE_DIR, "comprehensive_model_comparison_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_results, f, indent=4, ensure_ascii=False, default=str)

    print(f"\n✅ Comprehensive results saved to: {output_file}")

    # Print JSON structure preview
    print(f"\n📋 JSON file structure preview:")
    print(f"{{")
    for chip_name in list(comprehensive_results.keys())[:2]:  # Show only first 2 chips as example
        print(f'  "{chip_name}": {{')
        chip_data = comprehensive_results[chip_name]

        if 'simulation_model' in chip_data:
            print(f'    "simulation_model": {{')
            print(
                f'      "X_basis": {{ "5": {chip_data["simulation_model"]["X_basis"].get(5, "N/A"):.6f}, "9": ..., "13": ..., "17": ..., "21": ... }},')
            print(
                f'      "Z_basis": {{ "5": {chip_data["simulation_model"]["Z_basis"].get(5, "N/A"):.6f}, "9": ..., "13": ..., "17": ..., "21": ... }},')
            print(f'      "total_l1_X": {chip_data["simulation_model"]["total_l1_X"]:.6f},')
            print(f'      "total_l1_Z": {chip_data["simulation_model"]["total_l1_Z"]:.6f},')
            print(f'      "total_l1": {chip_data["simulation_model"]["total_l1"]:.6f}')
            print(f'    }},')

        for model_name in list(TRADITIONAL_MODELS.keys())[:2]:  # Show only first 2 traditional models
            if model_name in chip_data:
                model_data = chip_data[model_name]
                print(f'    "{model_name}": {{')
                print(f'      "p_value": {model_data["p_value"]:.6f},')
                print(
                    f'      "X_basis": {{ "5": {model_data["X_basis"].get(5, "N/A"):.6f}, "9": ..., "13": ..., "17": ..., "21": ... }},')
                print(
                    f'      "Z_basis": {{ "5": {model_data["Z_basis"].get(5, "N/A"):.6f}, "9": ..., "13": ..., "17": ..., "21": ... }},')
                print(f'      "total_l1_X": {model_data["total_l1_X"]:.6f},')
                print(f'      "total_l1_Z": {model_data["total_l1_Z"]:.6f},')
                print(f'      "total_l1": {model_data["total_l1"]:.6f}')
                print(f'    }},')

        print(f'    ...(other traditional models)')
        print(f'  }},')

    print(f'  ...(other chips)')
    print(f'}}')

    return output_file


if __name__ == "__main__":
    try:
        print("🚀 Starting model comparison analysis...")

        # Calculate optimized parameter models
        print("\n" + "=" * 60)
        print("Part 1: Optimized Parameter Model Analysis")
        print("=" * 60)
        all_chip_results = calculate_all_chips_l1_distances()
        print(f"\n🎯 Optimized parameter model analysis completed!")

        # Calculate traditional model vs real chip comparisons
        print("\n" + "=" * 60)
        print("Part 2: Traditional Model vs Real Chip Comparison Analysis")
        print("=" * 60)
        all_comparison_results = calculate_all_traditional_vs_chip_comparisons()
        print(f"\n🎯 Traditional model comparison analysis completed!")

        # Save comprehensive results
        print("\n" + "=" * 60)
        print("Part 3: Save Comprehensive Results")
        print("=" * 60)
        output_file = save_comprehensive_results(all_chip_results, all_comparison_results)

        print(f"\n🏆 All analysis completed!")
        print(f"📄 Comprehensive results file: {output_file}")

    except Exception as e:
        print(f"❌ Error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()