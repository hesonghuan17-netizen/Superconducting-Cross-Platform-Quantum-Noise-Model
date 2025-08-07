import qiskit.qasm2
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from generate_openqasm import generate_openqasm
import time
import sys
import os
from config import NUM_QUBITS, NUM_ROUNDS, TOKEN, LAYOUT, SHOTS_EXP

# Setup paths
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

# Initialize service and backend
service = QiskitRuntimeService(channel='ibm_cloud', token=TOKEN, instance='rep_code')
backend = service.backend("ibm_torino")

# Generate and prepare circuit
program = generate_openqasm(NUM_QUBITS, NUM_ROUNDS)
circuit = qiskit.qasm2.loads(program)
isa_circuit = transpile(circuit, backend=backend, initial_layout=LAYOUT)
q_layout = {v._index: k for v, k in isa_circuit.layout.initial_layout.get_virtual_bits().items() if
            v._register.name == 'q'}
q_layout = dict(sorted(q_layout.items()))


def run_job(sampler, circuit):
    """Function to run a single job."""
    job = sampler.run([circuit], shots=SHOTS_EXP)
    print(f">>> Job ID: {job.job_id()}")
    print(f">>> Job Status: {job.status()}")
    return job


def main():
    sampler = Sampler(mode=backend)

    # Create a list of circuits to run
    circuits = [isa_circuit for _ in range(24)]  # Assuming isa_circuit is the circuit to be run

    # Run jobs sequentially with delay
    for circuit in circuits:
        try:
            job = run_job(sampler, circuit)
            print(f"Job {job.job_id()} submitted with status {job.status()}")
            time.sleep(0.1)  # Add delay between submissions
        except Exception as e:
            print(f"Job failed with exception: {e}")


if __name__ == "__main__":
    main()

