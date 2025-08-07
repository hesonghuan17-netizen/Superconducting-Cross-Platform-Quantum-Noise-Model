import numpy as np
import math
import sys
import os



root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)




current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


def cal_rough_czfid(sqg_fid, ecr_fid):
    czfid = []  # Initialize czfid as an empty list
    for i in range(len(ecr_fid)):  # Ensure the loop iterates over the correct range
        #print(i)
        dqg_fid = sqg_fid[i]**2 * sqg_fid[i+1]**2
        new_value = (dqg_fid * ecr_fid[i])  # Calculate the new value
        czfid.append(new_value)  # Append the new value to czfid
    return czfid



def calculate_depolarizing_error_probability(dim, F_E_relax, F):
    if dim * F_E_relax == 1:
        raise ValueError("Division by zero error: ensure that `dim * F_E_relax` is not 1.")

    p = dim * (F_E_relax - F) / (dim * F_E_relax - 1)

    if p <= 0:
        p = 0

    return p


def calculate_pad_ppd(t, T1, T2):
    PAD = 1 - math.exp(-t / T1)
    PPD = 1 - (math.exp(-t / T2)**2 / (1 - PAD))
    return PAD, PPD

def calculate_fidelity(PAD, PPD):
    term1 = 1/3
    term2 = 1/3 * (1 - PAD) * (1 - PPD)
    term3 = 1/3 * math.sqrt((1 - PAD) * (1 - PPD))
    term4 = 1/6 * PAD
    term5 = 1/3 * (1 - PAD) * PPD
    fidelity = term1 + term2 + term3 + term4 + term5
    return fidelity

def calculate_decoherence_fidelity(t1_t2_values, t):
    fidelity_values = []

    for t1_value, t2_value in t1_t2_values:
        if t2_value > 2 * t1_value:
            t2_value = 2 * t1_value
        pad, ppd = calculate_pad_ppd(t, t1_value, t2_value)
        fidelity = calculate_fidelity(pad, ppd)
        fidelity_values.append(fidelity)

    return fidelity_values





