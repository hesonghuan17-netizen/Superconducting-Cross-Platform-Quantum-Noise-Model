def generate_openqasm(num_qubits, num_rounds):
    # Check if the number of qubits is odd
    if num_qubits % 2 == 0:
        raise ValueError("The number of qubits must be odd to match the required gate pattern.")

    # Ensure there are enough qubits to perform the specified operations
    if num_qubits < 5:
        raise ValueError("Minimum of 5 qubits required for the operations in this script.")

    qasm_script = "OPENQASM 2.0;\n"
    qasm_script += "include \"qelib1.inc\";\n"
    qasm_script += f"qreg q[{num_qubits}];\n"
    qasm_script += f"creg meas[{num_rounds * (num_qubits // 2)}];\n"  # One measurement per odd qubit per round

    # Create a single barrier command for all qubits
    all_qubits_barrier = ",".join(f"q[{i}]" for i in range(num_qubits))

    for round_num in range(num_rounds):
        # Generate CX gates for pairs (q[i], q[i+1]) where i starts from 0 and goes up to num_qubits-3
        for i in range(0, num_qubits - 2, 2):
            qasm_script += f"h q[{i+1}];\n"
            
        # Apply barrier to all qubits
        qasm_script += f"barrier {all_qubits_barrier};\n"    
        for i in range(0, num_qubits - 2, 2):
            qasm_script += f"cz q[{i}],q[{i+1}];\n"
            
        
        # Apply barrier to all qubits
        qasm_script += f"barrier {all_qubits_barrier};\n"
        
        # Generate CX gates for pairs (q[i+1], q[i]) where i starts from 2 and goes up to num_qubits-1
        for i in range(2, num_qubits, 2):
            qasm_script += f"cz q[{i}],q[{i-1}];\n"
        # Apply barrier to all qubits
        qasm_script += f"barrier {all_qubits_barrier};\n"
        for i in range(2, num_qubits, 2):
            qasm_script += f"h q[{i-1}];\n"

        # Apply barrier to all qubits
        qasm_script += f"barrier {all_qubits_barrier};\n"

        # Measure all odd qubits first
        for i in range(1, num_qubits, 2):
            qasm_script += f"measure q[{i}] -> meas[{round_num * (num_qubits // 2) + i // 2}];\n"

        # Apply barrier before reset operations
        qasm_script += f"barrier {all_qubits_barrier};\n"

        # Reset all odd qubits
        for i in range(1, num_qubits, 2):
            qasm_script += f"reset q[{i}];\n"

        # Apply final barrier to all qubits
        qasm_script += f"barrier {all_qubits_barrier};\n"

    return qasm_script