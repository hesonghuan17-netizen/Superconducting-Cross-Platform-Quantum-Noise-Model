from datetime import datetime
from run import backend, q_layout
import os
import json

# Define data directory path
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)

# Set calibration time
#calibrations_time = datetime(year=2025, month=2, day=26, hour=11, minute=0, second=0)
calibrations_time = datetime(year=2025, month=7, day=16, hour=11, minute=27, second=51)
#properties = backend.properties()#datetime=calibrations_time
properties = backend.properties(datetime=calibrations_time)


# Initialize data structure for JSON output
quantum_data = {
    "spam_rates": [],
    "t1_t2_values": [],
    "ecr_lengths": [],
    "ecr_fid": [],
    "sqg_fid": [],
    "rd_length": [],
    "lp": [],  # Custom defined parameters
    "sp": []  # Custom defined parameters
}

# Collect qubit properties
qubit_properties = {}
for key, value in q_layout.items():
    qubit_props = properties.qubit_property(value)
    readout_error = properties.readout_error(value)

    # Directly get T1 and T2 times
    try:
        t1_time = properties.t1(value)
    except:
        t1_time = None

    try:
        t2_time = properties.t2(value)
    except:
        t2_time = None

    qubit_properties[value] = {
        'readout_error': readout_error,
        't1': t1_time,
        't2': t2_time,
        'other_properties': qubit_props
    }

# Fill spam_rates (readout error)
for i in range(len(q_layout)):
    qubit = q_layout[i]
    readout_error = properties.readout_error(qubit)
    quantum_data["spam_rates"].append(readout_error)

# Fill t1_t2_values
for i in range(len(q_layout)):
    qubit = q_layout[i]
    quantum_data["t1_t2_values"].append({
        "t1": qubit_properties[qubit]['t1'],
        "t2": qubit_properties[qubit]['t2']
    })

# Fill ECR gate properties
ecr_lengths = []
ecr_fidelities = []

for i in range(len(q_layout) - 1):
    qubits_forward = [q_layout[i], q_layout[i + 1]]
    qubits_reverse = [q_layout[i + 1], q_layout[i]]

    try:
        # Try forward pairing
        ecr_length = properties.gate_length('cz', qubits_forward)
        ecr_err = properties.gate_error('cz', qubits_forward)
        ecr_fidelity = 1 - ecr_err  # Fidelity = 1 - error rate
        ecr_lengths.append(ecr_length)
        ecr_fidelities.append(ecr_fidelity)
    except Exception as e:
        # If forward fails, try reverse pairing
        try:
            ecr_length = properties.gate_length('cz', qubits_reverse)
            ecr_err = properties.gate_error('cz', qubits_reverse)
            ecr_fidelity = 1 - ecr_err
            ecr_lengths.append(ecr_length)
            ecr_fidelities.append(ecr_fidelity)
        except Exception as e2:
            # If both fail, fill with None or default values
            ecr_lengths.append(None)
            ecr_fidelities.append(None)

quantum_data["ecr_lengths"] = ecr_lengths
quantum_data["ecr_fid"] = ecr_fidelities

# Fill single qubit gate fidelity
for i in range(len(q_layout)):
    qubit = q_layout[i]
    sqg_err = properties.gate_error('x', [qubit])
    sqg_fidelity = 1 - sqg_err  # Fidelity = 1 - error rate
    quantum_data["sqg_fid"].append(sqg_fidelity)

# Fill readout length
for i in range(len(q_layout)):
    qubit = q_layout[i]
    rd_length = properties.readout_length(qubit)
    quantum_data["rd_length"].append(rd_length)

# Custom lp and sp parameters, all set to 0, ensuring 21 elements
quantum_data["lp"] = [0] * 21
quantum_data["sp"] = [0] * 21

# Save as JSON file
json_filename = os.path.join(data_dir, f'torino_layout_1_quantum_properties_{calibrations_time.strftime("%Y%m%d_%H%M%S")}.json')
with open(json_filename, 'w') as json_file:
    json.dump(quantum_data, json_file, indent=4)

print(f"Quantum device properties saved to: {json_filename}")