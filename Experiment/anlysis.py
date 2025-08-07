from qiskit_ibm_runtime import QiskitRuntimeService
import numpy as np
import sys
import os

# Get project root directory path and add to sys.path
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from config import NUM_MEA, NUM_ROUNDS, NUM_QUBITS, TOKEN


def analyze_job_results(job_id):
    """
    Analyze quantum job results and return the final array of detection events.

    Args:
        job_id (str): The ID of the quantum job to analyze

    Returns:
        np.ndarray: Array of detection events
    """
    service = QiskitRuntimeService(
        channel='ibm_cloud',
        instance='rep_code',
        token=TOKEN
    )
    job = service.job(job_id)
    job_result = job.result()

    pub_result = job_result[0]
    counts = pub_result.data.meas.get_counts()

    # Process the data
    reorganized_data = reorganize_data(counts, NUM_QUBITS)
    experiment_data = reorganized_data
    detection_results = calculate_detection_events(experiment_data)

    # Convert results to final array
    all_arrays = []
    for key, value in detection_results.items():
        interleaved = interleave_bits(*value[0])
        num_repeats = value[1]
        array = np.array([int(bit) for bit in interleaved], dtype=bool)
        repeated_array = np.tile(array, (num_repeats, 1))
        all_arrays.append(repeated_array)

    final_array = np.vstack(all_arrays)
    return np.array(final_array)


def reorganize_data(counts, num_qubits):
    experiment_data = {}

    # Calculating the step size
    step_size = (num_qubits - 1) // 2 if num_qubits > 1 else 1

    for binary_string, count in counts.items():
        # Length of the binary string
        length = len(binary_string)

        # Initialize a list to collect qubits
        qubits = []

        # Collect qubits based on step size
        for i in range(step_size):
            qubit = binary_string[length - 1 - i::-step_size]
            qubits.append(qubit)
        # Create a tuple of the collected qubits
        experiment_tuple = tuple(qubits)

        # Update the dictionary
        if experiment_tuple in experiment_data:
            experiment_data[experiment_tuple] += count
        else:
            experiment_data[experiment_tuple] = count

    return experiment_data


def calculate_detection_events(qubit_results):
    detection_events = {}

    for qubit_result, count in qubit_results.items():
        events = []
        for result in qubit_result:
            # Compare first bit with 0
            event = '1' if result[0] != '0' else '0'
            # Compare each subsequent bit with the previous bit
            event += ''.join('1' if result[i] != result[i - 1] else '0' for i in range(1, len(result)))
            events.append(event)
        detection_events[qubit_result] = (events, count)

    return detection_events


def calculate_detection_event_fraction(qubit_results):
    # Calculate total number of experiments
    total_experiments = sum(count for _, count in qubit_results.items())
    print(total_experiments)

    # Initialize event count for each round
    round_event_counts = [0] * NUM_ROUNDS

    for qubit_result, count in qubit_results.items():
        detection_events = calculate_detection_events({qubit_result: count})
        # print(detection_events)
        # Accumulate events for each qubit
        for events, event_count in detection_events.values():
            # print(events)
            for qubit_event in events:
                # Accumulate detection events for corresponding rounds bit by bit
                # print(qubit_event)
                for i in range(len(qubit_event)):
                    if qubit_event[i] == '1':
                        # print(event_count)
                        round_event_counts[i] += event_count

    # Calculate event fraction for each round
    event_fractions = [round(count / (NUM_MEA * total_experiments), 4) for count in round_event_counts]
    return event_fractions


def interleave_bits(*strings):
    # Find the maximum length of all strings
    max_len = max(len(s) for s in strings)

    # Pad each string to the maximum length with '0'
    padded_strings = [s.ljust(max_len, '0') for s in strings]

    # Interleave bits
    result = ''
    for bits in zip(*padded_strings):  # Unpack each string and zip them together
        result += ''.join(bits)  # Join bits from each string in order

    return result