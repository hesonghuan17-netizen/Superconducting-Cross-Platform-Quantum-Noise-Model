import stim
import random
import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)



from Simulation.Paramaeter_Loading.calculate_p import calculate_p_values, calculate_px_py_pz, calculate_lut, calculate_px_py_pz_rd

def leakage_and_seepage(c, count, l_p, s_p, qubit_leakage, index):
    """
    Appends 'RESET' and 'DEPOLARIZE' to the list c with probability p.
    Sets the flag qubit_leakage to True if these operations are performed.

    Parameters:
    - c (list): List to which operations will be appended.
    - p (float): Probability with which the operations are added.
    - qubit_leakage (dict): Dictionary tracking leakage status of qubits.

    Returns:
    - None
    """
    

    if qubit_leakage.get(index, False):
        if random.random() < s_p:
            qubit_leakage[index] = False 
        leak = stim.Circuit()
        leak.append("R", index)
        leak.append("DEPOLARIZE1", index, 0.75)
        c.insert(count, leak)
        count += 2

    if not qubit_leakage.get(index, False) and random.random() < l_p:
        leak = stim.Circuit()
        leak.append("R", index)
        leak.append("DEPOLARIZE1", index, 0.75)
        c.insert(count, leak)
        count += 2
        qubit_leakage[index] = True
        #print(index)
        
    return count

def apply_leakage_for_2qg(c, count, qubit_leakage, index1, index2):
    """
    Checks if there's leakage on either of the two specified qubits.
    If one of the qubits has leakage, it appends 'RESET' and 'DEPOLARIZE' operations
    to the other qubit.

    Parameters:
    - c (list): List to which operations will be appended.
    - qubit_leakage (dict): Dictionary tracking leakage status of qubits.
    - index1 (int): Index of the first qubit to check.
    - index2 (int): Index of the second qubit to check.

    Returns:
    - None
    """
    # Check if either of the qubits has leakage
    leak = stim.Circuit()
    if qubit_leakage.get(index1, False) and not qubit_leakage.get(index2, False):
        # If index1 has leakage, apply operations to index2
        
        leak.append("R", index2)
        leak.append("DEPOLARIZE1", index2, 0.75)
        c.insert(count, leak)
        count += 2
        #print(count)
        return count

    elif qubit_leakage.get(index2, False) and not qubit_leakage.get(index1, False):
        # If index2 has leakage, apply operations to index1
        leak.append("R", index1)
        leak.append("DEPOLARIZE1", index1, 0.75)
        c.insert(count, leak)
        count += 2
       #print(count)
        return count
    else:
        
        return count
    
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

def apply_2qg1(c, index1, index2, p1, p, lut, px_py_pz):
    """
    Parameters:
    - c (list): List to which operations will be appended.
    - index1 (int): Index of the first qubit to check.
    - index2 (int): Index of the second qubit to check.

    Returns:
    - None
    """
    c.append("H", index2)
    c.append("PAULI_CHANNEL_1", index2, px_py_pz[index2])
    c.append("DEPOLARIZE1", index2, p1[index2])
    

    c.append("CZ", [index1, index2])
    c.append("PAULI_CHANNEL_1", index1, lut[index1][index1])
    c.append("PAULI_CHANNEL_1", index2, lut[index1][index2])
    c.append("DEPOLARIZE2", [index1, index2], p[index1])

def simulate_leakage_operations(c, rounds, count, d_count2, d_count3, d_count4, num_qubits, lp, sp, qubit_leakage, spam_rates):

    for _ in range(rounds):

        # Reset specified qubits' leakage status for every round
        for i in range(0, num_qubits-1, 2):
            if i < num_qubits:
                qubit_leakage[i+1] = False

        # Apply leakage to pairs of qubits
        for i in range(0, num_qubits - 1, 2):
            count = apply_leakage_for_2qg(c, count, qubit_leakage, i, i+1)
        
        # Apply leakage and seepage effects to individual qubits
        for i in range(num_qubits-1):
            count = leakage_and_seepage(c, count, lp[i], sp[i], qubit_leakage, i)
        #print(qubit_leakage)
        #print(count)
        count += d_count4
        # Apply leakage to other pairs, dynamically wrapping around if needed
        for i in range(1, num_qubits, 2):
            count = apply_leakage_for_2qg(c, count, qubit_leakage, i+1, i)

        # Apply leakage and seepage effects again
        for i in range(1, num_qubits):
            count = leakage_and_seepage(c, count, lp[i], sp[i], qubit_leakage, i)
        count += d_count3 + d_count2

    return count 

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

def simulate_operations(rounds, num_qubits, spam_rates, p1, p, lut, px_py_pz, px_py_pz_rd):
    # List to store the results_test from each iteration
    circuits = []

    for _ in range(rounds):
        # Create a new circuit instance for each iteration
        cr = stim.Circuit()

        # Apply leakage to pairs of qubits
        for i in range(0, num_qubits - 1, 2):
            apply_2qg1(cr, i, i+1, p1, p, lut, px_py_pz)
        count1_cr = len(cr)
        # Apply leakage to other pairs, dynamically wrapping around if needed
        for i in range(1, num_qubits, 2):
            apply_2qg2(cr, i+1, i, p1, p, lut, px_py_pz)
        count2_cr = len(cr)
        # Stimulate errors and measure all qubits
        # Create a string of space-separated indices for odd-numbered qubits only
        error_instr = ' '.join([str(i) for i in range(1, num_qubits, 2)])

        stim_program_text = generate_stim_program(num_qubits, spam_rates)
        # Use this string in the Stim program text to apply operations only to odd-numbered qubits
        cr.append_from_stim_program_text(stim_program_text)
        cr.append_from_stim_program_text(f'''
        MR {error_instr}
        ''')


        # Add detectors for error correction based on measurements
        for i in range((num_qubits-1) // 2, 0, -1):
                cr.append_from_stim_program_text(f'''
                DETECTOR rec[-{i}] rec[-{i+((num_qubits-1)//2)}]
                ''')
        for i in range(0, num_qubits, 2):
            cr.append("PAULI_CHANNEL_1", i, px_py_pz_rd[i])

        # Append the circuit result to the results_test list
        circuits.append(cr)
    d_count2 = count1_cr
    d_count3 = count2_cr - count1_cr

    return d_count2, d_count3, circuits

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

def generate_prefix(num_qubits, p1, p, lut, px_py_pz):
    # Check if the number of qubits is odd
    if num_qubits % 2 == 0:
        return "Error: The number of qubits must be odd."

    # Begin the script with the number of qubits and depolarizing rate
    script = f"# Number of qubits: {num_qubits}\n"

    # Add rotation gates for each qubit
    script += "R " + " ".join(str(i) for i in range(num_qubits)) + "\n"

    # Add H gates
    script += "H " + " ".join(str(i) for i in range(1, num_qubits, 2)) + "\n"

    for i in range(1, num_qubits, 2):
        # 修复：添加括号
        script += f"PAULI_CHANNEL_1({px_py_pz[i][0]}, {px_py_pz[i][1]}, {px_py_pz[i][2]}) {i}\n"
        script += f"DEPOLARIZE1({p1[i]}) {i}\n"

        # Add CNOT gates from qubit 0 to qubit n-1 sequentially
    script += "CZ " + " ".join(str(i) for i in range(num_qubits - 1)) + "\n"

    for i in range(1, num_qubits, 2):
        # 修复：添加括号
        script += f"PAULI_CHANNEL_1({lut[i - 1][i - 1][0]}, {lut[i - 1][i - 1][1]}, {lut[i - 1][i - 1][2]}) {i - 1}\n"
        script += f"PAULI_CHANNEL_1({lut[i - 1][i][0]}, {lut[i - 1][i][1]}, {lut[i - 1][i][2]}) {i}\n"
        script += f"DEPOLARIZE2({p[i - 1]}) {i - 1} {i}\n"

    return script

def generate_circuits(SHOTS, rounds, num_qubits, lp, sp, spam_rates, sqg_fid, ecr_fid, t1_t2_values, ecr_lengths, rd_lengths, sqg_lengths):

    p1, p = calculate_p_values(sqg_fid, ecr_fid, t1_t2_values, ecr_lengths, sqg_lengths)
    lut = calculate_lut(t1_t2_values, ecr_lengths, sqg_lengths)
    px_py_pz = calculate_px_py_pz(t1_t2_values, sqg_lengths)
    px_py_pz_rd = calculate_px_py_pz_rd(t1_t2_values, rd_lengths)

    circuits = []
    prefix = generate_prefix(num_qubits, p1, p, lut, px_py_pz)
    stim_program_text = generate_stim_program(num_qubits, spam_rates)
    stim_program_text_end = generate_stim_program_end(num_qubits, spam_rates)
    error_instr1 = ' '.join([str(i) for i in range(1, num_qubits, 2)])
    c = stim.Circuit()
    c.append_from_stim_program_text(prefix)
    count1 = len(c)
    #c.insert(count1, stim.CircuitInstruction("Y", [3, 4, 5]))
    for i in range(1, num_qubits, 2):
            apply_2qg2(c, i+1, i, p1, p, lut, px_py_pz)
    count2 = len(c)
    d_count1 = count2 - count1
    #print(len(c))
    c.append_from_stim_program_text(stim_program_text)      
    c.append_from_stim_program_text(f'''
    MR {error_instr1}
    ''')    
    # Add detectors for error correction based on measurements
    for i in range((num_qubits-1) // 2, 0, -1):
        c.append_from_stim_program_text(f'''
        DETECTOR rec[-{i}] 
        ''')
    #start_time = timeit.default_timer()

    for i in range(0, num_qubits, 2):
        c.append("PAULI_CHANNEL_1", i, px_py_pz_rd[i])    
    count3 = len(c)
    d_count2 = count3 - count2
    #print(d_count1)
    d_count3, d_count4, Cr = simulate_operations(rounds, num_qubits, spam_rates, p1, p, lut, px_py_pz, px_py_pz_rd) 
    #print(d_count3)
    #print(d_count4)
    for i in range(rounds):
        c += Cr[i] 
    error_instr2 = ' '.join([str(i) for i in range(0, num_qubits, 2)])

        # Use this string in the Stim program text to apply operations only to odd-numbered qubits
    c.append_from_stim_program_text(stim_program_text_end)        
    c.append_from_stim_program_text(f'''
    MR {error_instr2}
    ''')

    c.append_from_stim_program_text('''
    OBSERVABLE_INCLUDE(0) rec[-1]
    ''')
    
    for _ in range(SHOTS):
        
        c_new = c.copy()
        
        qubit_leakage = {}
        # Assuming qubit_leakage is already defined and contains some entries
        for i in range(0, num_qubits):
            if i < num_qubits:
                qubit_leakage[i] = False
        #print(qubit_leakage)
                
        count = count1    
        
        for i in range(num_qubits-1):
            count = leakage_and_seepage(c_new, count, lp[i], sp[i], qubit_leakage, i)
        #print(qubit_leakage)
            
        count += d_count1
        
        # Apply leakage to other pairs, dynamically wrapping around if needed
        for i in range(1, num_qubits, 2):
            count = apply_leakage_for_2qg(c_new, count, qubit_leakage, i+1, i)
        #print(count)
        
        
        for i in range(1, num_qubits):
            count = leakage_and_seepage(c_new, count, lp[i], sp[i], qubit_leakage, i)
        
        count += d_count2
        count += d_count3
        #c_new.insert(count, stim.CircuitInstruction("Y", [3, 4, 5]))
        #print(count)
        #start_time = timeit.default_timer()
        count = simulate_leakage_operations(c_new, rounds,  count, d_count2, d_count3, d_count4, num_qubits, lp, sp, qubit_leakage, spam_rates) 
        #elapsed = timeit.default_timer() -start_time
        #print(elapsed)
        circuits.append(c_new)
        
       
    return circuits

