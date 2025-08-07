import numpy as np
import sys
import os



current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from operations_simulation import generate_circuits


def run_sampling(shots,shots2, rounds, num_qubits, lp, sp, spam_rates, sqg_fid, ecr_fid, t1_t2_values, ecr_lengths, rd_lengths, sqg_lengths):
    """
    Generates quantum circuits based on parameters, compiles them,
    and samples measurement results_test, returning a 2D array of results_test.

    Parameters:
    - shots: Number of measurement shots per circuit.
    - rounds: Number of measurement rounds for each circuit.
    - num_qubits: Number of qubits in each circuit.
    - lp: List of probabilities associated with certain errors or characteristics.
    - sp: List of secondary probabilities for different conditions.
    - spam_rates: List of SPAM (State Preparation and Measurement) error rates.

    Returns:
    - A 2D numpy array where each row represents the sampled results_test from one circuit.
    """
    results = []
    circuit_list = generate_circuits(shots, rounds, num_qubits, lp, sp, spam_rates, sqg_fid, ecr_fid, t1_t2_values, ecr_lengths, rd_lengths, sqg_lengths)

    for circuit in circuit_list:
        sampler = circuit.compile_detector_sampler()
        # Sample shots and ensure it's reshaped if necessary
        res_det = sampler.sample(shots=shots2).reshape(-1)  # Reshaping to a 1D array per sample
        results.append(res_det)

    # Convert list of 1D arrays to a 2D NumPy array
    results_array = np.array(results)
    return results_array

