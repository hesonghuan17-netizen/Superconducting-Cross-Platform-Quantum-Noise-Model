import os
import sys
import json
import numpy as np

# Get project root directory and add to path
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)

from config import NUM_QUBITS
from Simulation.Paramaeter_Loading.main import (
    calculate_depolarizing_error_probability,
    calculate_decoherence_fidelity,
    cal_rough_czfid
)



def calculate_p_values(sqg_fid, ecr_fid, t1_t2_values, ecr_lengths, t):
    """
    Calculate depolarizing error probabilities using given fidelities.
    
    Args:
        sqg_fid (list): Single-qubit gate fidelities
        ecr_fid (list): ECR gate fidelities
        t (float, optional): Time parameter for calculations
        parameters (dict, optional): Dictionary containing other required parameters.
                                   If None, loads from default JSON file.
    
    Returns:
        tuple: (p1, p) where:
            - p1: dict of single-qubit depolarizing error probabilities
            - p: dict of two-qubit depolarizing error probabilities
    """

    
    # Calculate CZ fidelities
    czfid = cal_rough_czfid(sqg_fid, ecr_fid)
    
    # Calculate decoherence fidelities
    sqg_dec_f = calculate_decoherence_fidelity(t1_t2_values, t)
    # Calculate lookup table fidelities
    lut_f = {i: {} for i in range(NUM_QUBITS)}
    for i in range(NUM_QUBITS-1):
        #print(i)
        current_t = 2 * t + ecr_lengths[i]
        infs = calculate_decoherence_fidelity(t1_t2_values, current_t)
        for j in range(len(t1_t2_values)):
            lut_f[i][j] = infs[j]
    
    # Calculate single-qubit depolarizing probabilities
    dim1 = 2**1
    p1 = {}
    for i in range(NUM_QUBITS):
        F_relax = sqg_dec_f[i]
        p1[i] = calculate_depolarizing_error_probability(dim1, F_relax, sqg_fid[i])
    
    # Calculate two-qubit depolarizing probabilities
    dim2 = 2**2
    p = {}
    for i in range(NUM_QUBITS-1):
        F_relax = lut_f[i][i] * lut_f[i][i+1]
        p[i] = calculate_depolarizing_error_probability(dim2, F_relax, czfid[i])
    
    return p1, p

def print_p_values(p1, p):
    """Pretty print the p values"""
    print("\nSingle-qubit depolarizing probabilities (p1):")
    for qubit, prob in p1.items():
        print(f"Qubit {qubit}: {prob:.6f}")
    
    print("\nTwo-qubit depolarizing probabilities (p):")
    for qubits, prob in p.items():
        print(f"Qubits {qubits}-{qubits+1}: {prob:.6f}")

def calculate_px_py_pz(t1_t2_values, t):
    """
    Calculate px, py, pz values for a single qubit given T1, T2, and time t.
    
    Args:
        t1 (float): T1 relaxation time
        t2 (float): T2 dephasing time
        t (float): Time duration
        
    Returns:
        tuple: (px, py, pz) values
    """
    # Ensure T2 ≤ 2T1 physical constraint
    px_py_pz_values = []

    for t1_value, t2_value in t1_t2_values:
           
        if t2_value > 2 * t1_value:
            t2_value = 2 * t1_value
        
        px_py = (1 - np.exp(-t / t1_value)) / 4
        pz = (1 - np.exp(-t / t2_value)) / 2 - (1 - np.exp(-t / t1_value)) / 4
        px_py_pz_values.append((px_py, px_py, pz))

    return px_py_pz_values

def calculate_lut(t1_t2_values, ecr_lengths, t):
    """
    Calculate lookup table (lut) for px, py, pz values for each qubit pair.
    
    Args:
        t1_t2_values: List of tuples [(t1, t2), ...] containing T1 and T2 values
        t: Base time duration for single-qubit gates
        ecr_lengths: List of ECR gate lengths
        
    Returns:
        dict: lut containing px, py, pz values for each qubit pair
    """
    num_qubits = len(t1_t2_values)
    lut = {i: {} for i in range(num_qubits - 1)}
    
    for i in range(num_qubits - 1):
        current_t = 2 * t + ecr_lengths[i]  # Total time including ECR gate
        px_py_pz_dqg = calculate_px_py_pz(t1_t2_values, current_t)
        for j in range(len(px_py_pz_dqg)):
            lut[i][j] = px_py_pz_dqg[j]
    
    return lut

def calculate_px_py_pz_rd(t1_t2_values, rd_lengths):
    """
    Calculate readout px, py, pz values for each qubit.
    
    Args:
        t1_t2_values: List of tuples [(t1, t2), ...] containing T1 and T2 values
        rd_lengths: List of readout lengths for each qubit
        
    Returns:
        dict: px_py_pz_rd containing readout px, py, pz values for each qubit
    """
    num_qubits = len(t1_t2_values)
    px_py_pz_rd = {}
    
    for i in range(num_qubits):
        px_py_pz_rd[i] = calculate_px_py_pz(t1_t2_values, rd_lengths[i])[i]
    
    return px_py_pz_rd

