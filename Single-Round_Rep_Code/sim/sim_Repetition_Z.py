import stim
import random
import sys
import os
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 获取当前文件的路径并添加项目根目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)




root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(root_path)

from Simulation.Paramaeter_Loading.calculate_p import calculate_p_values, calculate_px_py_pz, calculate_lut, calculate_px_py_pz_rd

    
def apply_2qg2(c, index1, index2, p1, p, lut, px_py_pz):
    """
    Parameters:
    - index1 (int): Index of the first qubit to check.
    - index2 (int): Index of the second qubit to check.

    Returns:
    - None
    """
    

    #start_time = timeit.default_timer()
    c.append("CZ", [index1, index2])
    #elapsed = timeit.default_timer() -start_time
    #print(elapsed)
    c.append("PAULI_CHANNEL_1", index1, lut[index2][index1])
    c.append("PAULI_CHANNEL_1", index2, lut[index2][index2])
    c.append("DEPOLARIZE2", [index1, index2], p[index2])
        
    c.append("H", index2)
    c.append("PAULI_CHANNEL_1", index2, px_py_pz[index2])
    c.append("DEPOLARIZE1", index2, p1[index2])

def generate_stim_program(num_qubits, spam_rates):
    """
    Generate Stim program text with distinct spam rates for odd-numbered qubits.

    :param num_qubits: Total number of qubits
    :param spam_rates: List of spam rates corresponding to each qubit
    :return: Stim program text as a single string
    """
    spam_lines = []

    # Ensure that there are enough spam rates provided
    if len(spam_rates) < num_qubits:
        raise ValueError("Insufficient number of spam rates provided.")

    for i in range(num_qubits):
        if i % 2 == 1:  # Check if the qubit number is odd
            spam_lines.append(f"X_ERROR({spam_rates[i]}) {i}")
    spam_text = '\n'.join(spam_lines)
    return spam_text

def generate_stim_program_end(num_qubits, spam_rates):
    """
    Generate Stim program text with distinct spam rates for odd-numbered qubits.

    :param num_qubits: Total number of qubits
    :param spam_rates: List of spam rates corresponding to each qubit
    :return: Stim program text as a single string
    """
    spam_lines = []

    # Ensure that there are enough spam rates provided
    if len(spam_rates) < num_qubits:
        raise ValueError("Insufficient number of spam rates provided.")

    for i in range(num_qubits):
        if i % 2 == 0:  # Check if the qubit number is even
            spam_lines.append(f"X_ERROR({spam_rates[i]}) {i}")
    spam_text = '\n'.join(spam_lines)
    return spam_text

def generate_stim_program_initial(num_qubits, spam_rates_initial):
    """
    Generate Stim program text with distinct spam rates for odd-numbered qubits.

    :param num_qubits: Total number of qubits
    :param spam_rates_initial: List of spam rates corresponding to each qubit
    :return: Stim program text as a single string
    """
    spam_lines = []

    # Ensure that there are enough spam rates provided
    if len(spam_rates_initial) < num_qubits:
        raise ValueError("Insufficient number of spam rates provided.")

    for i in range(num_qubits):
        spam_lines.append(f"X_ERROR({spam_rates_initial[i]}) {i}")
    spam_text = '\n'.join(spam_lines)
    return spam_text

def generate_prefix(num_qubits, p1, p, lut, px_py_pz):
    # Check if the number of qubits is odd
    if num_qubits % 2 == 0:
        return "Error: The number of qubits must be odd."

    # Begin the script with the number of qubits and depolarizing rate
    script = f"# Number of qubits: {num_qubits}\n"

    # Add H gates
    script += "H " + " ".join(str(i) for i in range(1, num_qubits, 2)) + "\n"

    for i in range(1, num_qubits, 2):
        script += f"PAULI_CHANNEL_1({px_py_pz[i][0]}, {px_py_pz[i][1]}, {px_py_pz[i][2]}) {i}\n"
        script += f"DEPOLARIZE1({p1[i]}) {i}\n"

    # Add CNOT gates from qubit 0 to qubit n-1 sequentially
    script += "CZ " + " ".join(str(i) for i in range(num_qubits - 1)) + "\n"

    for i in range(1, num_qubits, 2):
        script += f"PAULI_CHANNEL_1({lut[i - 1][i - 1][0]}, {lut[i - 1][i - 1][1]}, {lut[i - 1][i - 1][2]}) {i - 1}\n"
        script += f"PAULI_CHANNEL_1({lut[i - 1][i][0]}, {lut[i - 1][i][1]}, {lut[i - 1][i][2]}) {i}\n"
        # print(ecr_lengths[i-1])
        # Add DEPOLARIZE2 operation to each qubit from 0 to n-1 with the specific depolarizing rate
        script += f"DEPOLARIZE2({p[i - 1]}) {i - 1} {i}\n"

    return script

def generate_circuits(SHOTS, _rounds_ignored, num_qubits, lp, sp, spam_rates, spam_rates_initial, sqg_fid, ecr_fid, t1_t2_values, ecr_lengths,
                      rd_lengths, sqg_lengths):
    rounds = 1  # Force rounds = 1

    p1, p = calculate_p_values(sqg_fid, ecr_fid, t1_t2_values, ecr_lengths, sqg_lengths)
    lut = calculate_lut(t1_t2_values, ecr_lengths, sqg_lengths)
    px_py_pz = calculate_px_py_pz(t1_t2_values, sqg_lengths)
    px_py_pz_rd = calculate_px_py_pz_rd(t1_t2_values, rd_lengths)

    circuits = []
    prefix = generate_prefix(num_qubits, p1, p, lut, px_py_pz)
    stim_program_text_initial = generate_stim_program_initial(num_qubits, spam_rates_initial)
    stim_program_text = generate_stim_program(num_qubits, spam_rates)
    stim_program_text_end = generate_stim_program_end(num_qubits, spam_rates)
    # error_instr1 = ' '.join([str(i) for i in range(1, num_qubits, 2)])
    c = stim.Circuit()
    c.append_from_stim_program_text(stim_program_text_initial)
    c.append_from_stim_program_text(prefix)

    count1 = len(c)
    for i in range(1, num_qubits, 2):
        apply_2qg2(c, i + 1, i, p1, p, lut, px_py_pz)
    count2 = len(c)
    d_count1 = count2 - count1

    c.append_from_stim_program_text(stim_program_text)
    # c.append_from_stim_program_text(f'M {error_instr1}')  # 🔁 Replaced MR → M

    count3 = len(c)
    d_count2 = count3 - count2

    # error_instr2 = ' '.join([str(i) for i in range(0, num_qubits, 2)])
    c.append_from_stim_program_text(stim_program_text_end)
    # c.append_from_stim_program_text(f'M {error_instr2}')  # 🔁 Replaced MR → M\
    error_instr = ' '.join([str(i) for i in range(0, num_qubits)])
    c.append_from_stim_program_text(f'M {error_instr}')
    # Remove the final OBSERVABLE_INCLUDE
    # Do not add detectors or observables

    circuits.append(c)
    return circuits


def run_sampling(shots, shots2, rounds, num_qubits, lp, sp, spam_rates, spam_rates_initial, sqg_fid, ecr_fid, t1_t2_values, ecr_lengths,
                 rd_lengths, sqg_lengths):
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
    circuit_list = generate_circuits(shots, rounds, num_qubits, lp, sp, spam_rates, spam_rates_initial, sqg_fid, ecr_fid, t1_t2_values,
                                     ecr_lengths, rd_lengths, sqg_lengths)


    for circuit in circuit_list:
        # print(circuit.diagram())
        sampler = circuit.compile_sampler()
        # Sample shots and ensure it's reshaped if necessary
        res_det = sampler.sample(shots=shots2)  # Reshaping to a 1D array per sample
        results.append(res_det)

    # Convert list of 1D arrays to a 2D NumPy array
    results_array = np.array(results)
    results_array = results_array.reshape(-1, results_array.shape[2])
    return results_array

def analyze_results(results_array, result_dir, num_qubits, TIMES):
    os.makedirs(result_dir, exist_ok=True)
    avg_counts = {}
    total_shots = results_array.shape[1]

    for row in results_array:
        bitstring = ''.join(str(int(b)) for b in row)
        reversed_bitstring = bitstring[::-1]  # Reverse the bitstring
        if reversed_bitstring not in avg_counts:
            avg_counts[reversed_bitstring] = 0
        avg_counts[reversed_bitstring] += 1

    # Normalize by TIMES and round to int
    for state in avg_counts:
        avg_counts[state] = int(round(avg_counts[state] / TIMES))
    # Save counts to JSON
    json_path = os.path.join(result_dir, f"avg_counts_n{num_qubits}.json")
    with open(json_path, 'w') as f:
        json.dump(avg_counts, f, indent=2)

    # Plot counts
    percent_counts = {k: (v / (len(results_array)/TIMES)) * 100 for k, v in avg_counts.items()}
    sorted_counts = dict(sorted(percent_counts.items(), key=lambda item: item[1], reverse=True))
    top_5 = dict(list(sorted_counts.items())[:5])

    plt.figure(figsize=(12, 6))
    plt.bar(sorted_counts.keys(), sorted_counts.values(), color='blue')
    for state in top_5:
        plt.bar(state, sorted_counts[state], color='red')
    plt.xlabel('Reversed Measurement Outcomes')
    plt.ylabel('Percentage (%)')
    plt.title(f'Averaged Results for n={num_qubits}')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plot_path = os.path.join(result_dir, f"avg_plot_n{num_qubits}.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"✅ Analysis complete. Saved to {result_dir}")

