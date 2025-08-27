import numpy as np
import math


def calculate_depolarizing_error_probability(dim, F_E_relax, F):
    """
    计算去极化错误概率

    Args:
        dim: 维度 (2 for single qubit, 4 for two qubits)
        F_E_relax: 退相干保真度
        F: 门保真度

    Returns:
        float: 去极化错误概率
    """
    if dim * F_E_relax == 1:
        return 0  # 避免除零错误

    p = dim * (F_E_relax - F) / (dim * F_E_relax - 1)
    return max(0, p)  # 确保非负


def calculate_pad_ppd(t, T1, T2):
    """计算PAD和PPD"""
    PAD = 1 - math.exp(-t / T1)
    PPD = 1 - (math.exp(-t / T2) ** 2 / (1 - PAD)) if PAD != 1 else 0
    return PAD, PPD


def calculate_decoherence_fidelity_single(t1, t2, t):
    """
    计算单个量子比特的退相干保真度

    Args:
        t1: T1时间
        t2: T2时间
        t: 门执行时间

    Returns:
        float: 退相干保真度
    """
    # 确保物理约束 T2 <= 2*T1
    if t2 > 2 * t1:
        t2 = 2 * t1

    pad, ppd = calculate_pad_ppd(t, t1, t2)

    # 保真度计算公式
    term1 = 1 / 3
    term2 = 1 / 3 * (1 - pad) * (1 - ppd)
    term3 = 1 / 3 * math.sqrt((1 - pad) * (1 - ppd))
    term4 = 1 / 6 * pad
    term5 = 1 / 3 * (1 - pad) * ppd

    fidelity = term1 + term2 + term3 + term4 + term5
    return fidelity


def calculate_px_py_pz_single(t1, t2, t):
    """
    计算单个量子比特的Pauli通道概率

    Args:
        t1: T1时间
        t2: T2时间
        t: 门执行时间

    Returns:
        tuple: (px, py, pz)
    """
    # 确保物理约束
    if t2 > 2 * t1:
        t2 = 2 * t1

    px_py = (1 - np.exp(-t / t1)) / 4
    pz = (1 - np.exp(-t / t2)) / 2 - (1 - np.exp(-t / t1)) / 4

    return (px_py, px_py, pz)


def calculate_single_qubit_noise(qubit_params):
    """
    计算单量子比特门噪声参数

    Args:
        qubit_params: 单个量子比特的参数字典，包含：
            - t1, t2: 相干时间
            - sqg_fid: 单量子比特门保真度
            - sqg_length: 单量子比特门时间

    Returns:
        dict: 包含 'px_py_pz' 和 'p1' 的字典
    """
    t1 = qubit_params['t1']
    t2 = qubit_params['t2']
    sqg_fid = qubit_params['sqg_fid']
    sqg_length = qubit_params['sqg_length']

    # 计算退相干保真度
    F_relax = calculate_decoherence_fidelity_single(t1, t2, sqg_length)

    # 计算去极化概率
    p1 = calculate_depolarizing_error_probability(2, F_relax, sqg_fid)

    # 计算Pauli通道概率
    px_py_pz = calculate_px_py_pz_single(t1, t2, sqg_length)

    return {
        'px_py_pz': px_py_pz,
        'p1': p1
    }


def calculate_cx_fidelity(control_params, target_params, cx_fid):
    """
    计算CX门的总保真度

    Args:
        control_params: 控制量子比特参数
        target_params: 目标量子比特参数
        cx_fid: CX门的原生保真度

    Returns:
        float: CX门总保真度
    """
    # 包含的单量子比特门保真度（假设CX门实现需要一些单门操作）
    control_sqg_fid = control_params['sqg_fid']
    target_sqg_fid = target_params['sqg_fid']

    # 总保真度 = 单门保真度^2 * CX门保真度（类似原来的CZ门计算）
    total_fid = control_sqg_fid ** 2 * target_sqg_fid ** 2 * cx_fid

    return total_fid


def calculate_two_qubit_noise(control_params, target_params, cx_params):
    """
    计算双量子比特门噪声参数

    Args:
        control_params: 控制量子比特参数字典
        target_params: 目标量子比特参数字典
        cx_params: CX门参数字典，包含：
            - cx_fid: CX门保真度
            - cx_length: CX门时间

    Returns:
        dict: 包含各噪声参数的字典
    """
    cx_fid = cx_params['cx_fid']
    cx_length = cx_params['cx_length']

    # 获取量子比特参数
    control_t1 = control_params['t1']
    control_t2 = control_params['t2']
    control_sqg_length = control_params['sqg_length']

    target_t1 = target_params['t1']
    target_t2 = target_params['t2']
    target_sqg_length = target_params['sqg_length']

    # 计算CX门总时间（包含辅助单量子比特门）
    total_time = 2 * target_sqg_length + cx_length

    # 计算每个量子比特在总时间下的退相干保真度
    control_decoherence = calculate_decoherence_fidelity_single(control_t1, control_t2, total_time)
    target_decoherence = calculate_decoherence_fidelity_single(target_t1, target_t2, total_time)

    # 联合退相干保真度（独立假设）
    joint_decoherence = control_decoherence * target_decoherence

    # 计算CX门总保真度
    cx_total_fid = calculate_cx_fidelity(control_params, target_params, cx_fid)

    # 计算双量子比特去极化概率
    p2 = calculate_depolarizing_error_probability(4, joint_decoherence, cx_total_fid)

    # 计算各量子比特在总时间下的Pauli通道概率
    control_pauli = calculate_px_py_pz_single(control_t1, control_t2, total_time)
    target_pauli = calculate_px_py_pz_single(target_t1, target_t2, total_time)

    return {
        'control_pauli': control_pauli,
        'target_pauli': target_pauli,
        'p2': p2,
        'total_time': total_time
    }


def calculate_readout_noise(qubit_params):
    """
    计算读取时的噪声参数

    Args:
        qubit_params: 量子比特参数字典，包含：
            - t1, t2: 相干时间
            - rd_length: 读取时间

    Returns:
        tuple: (px, py, pz) 读取时的Pauli通道概率
    """
    t1 = qubit_params['t1']
    t2 = qubit_params['t2']
    rd_length = qubit_params['rd_length']

    return calculate_px_py_pz_single(t1, t2, rd_length)


# 方便使用的包装函数
def get_single_gate_noise_from_json(qubit_id, params_dict):
    """
    从JSON参数字典中获取单量子比特门噪声

    Args:
        qubit_id: 量子比特ID（字符串）
        params_dict: 完整的参数字典

    Returns:
        dict: 噪声参数
    """
    qubit_params = params_dict['qubits'][str(qubit_id)]
    return calculate_single_qubit_noise(qubit_params)


def get_two_gate_noise_from_json(cx_gate_id, params_dict):
    """
    从JSON参数字典中获取双量子比特门噪声

    Args:
        cx_gate_id: CX门ID（字符串）
        params_dict: 完整的参数字典

    Returns:
        dict: 噪声参数
    """
    cx_params = params_dict['cx_gates'][str(cx_gate_id)]
    control_id = cx_params['control']
    target_id = cx_params['target']

    control_params = params_dict['qubits'][str(control_id)]
    target_params = params_dict['qubits'][str(target_id)]

    return calculate_two_qubit_noise(control_params, target_params, cx_params)


def get_readout_idle_noise_from_json(qubit_id, params_dict):
    """
    从JSON参数字典中获取读取噪声

    Args:
        qubit_id: 量子比特ID（字符串）
        params_dict: 完整的参数字典

    Returns:
        tuple: (px, py, pz)
    """
    qubit_params = params_dict['qubits'][str(qubit_id)]
    return calculate_readout_noise(qubit_params)