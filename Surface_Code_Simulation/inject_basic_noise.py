import stim
import json
from calculate import get_single_gate_noise_from_json, get_readout_idle_noise_from_json, get_two_gate_noise_from_json


def load_surface_code_params(json_file):
    """
    加载表面码参数JSON文件

    Args:
    - json_file: JSON文件路径

    Returns:
    - params_dict: 参数字典
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        params_dict = json.load(f)
    return params_dict


def add_data_init_noise(circuit, data_qubits, params):
    """
    添加数据比特初始化噪声（R门后的X_ERROR）

    Args:
    - circuit: stim.Circuit对象
    - data_qubits: 数据比特列表
    - params: 参数字典
    """
    for qubit in data_qubits:
        qubit_params = params["qubits"][str(qubit)]
        error_rate = qubit_params["data_init_error"]
        if error_rate > 0:
            circuit.append("X_ERROR", qubit, error_rate)


def add_data_measurement_noise(circuit, data_qubits, params):
    """
    添加数据比特测量噪声（M门前的X_ERROR）

    Args:
    - circuit: stim.Circuit对象
    - data_qubits: 数据比特列表
    - params: 参数字典
    """
    for qubit in data_qubits:
        qubit_params = params["qubits"][str(qubit)]
        error_rate = qubit_params["data_measurement_error"]
        if error_rate > 0:
            circuit.append("X_ERROR", qubit, error_rate)


def add_readout_idle_noise(circuit, data_qubits, params):
    """
    添加数据比特在测量期间的空闲噪声

    Args:
    - circuit: stim.Circuit对象
    - data_qubits: 数据比特列表
    - params: 参数字典
    """

    for qubit in data_qubits:
        readout_noise = get_readout_idle_noise_from_json(qubit, params)
        circuit.append("PAULI_CHANNEL_1", qubit, readout_noise)


def add_measurement_spam_noise(circuit, measurement_qubits, params):
    """
    添加测量比特SPAM噪声（MR前的X_ERROR）

    Args:
    - circuit: stim.Circuit对象
    - measurement_qubits: 测量比特列表
    - params: 参数字典
    """
    for qubit in measurement_qubits:
        qubit_params = params["qubits"][str(qubit)]
        error_rate = qubit_params["measurement_spam_rate"]
        if error_rate > 0:
            circuit.append("X_ERROR", qubit, error_rate)


def add_single_gate_noise(circuit, qubit, params):
    """
    添加单量子比特门噪声：PAULI_CHANNEL_1 + DEPOLARIZE1

    Args:
    - circuit: stim.Circuit对象
    - qubit: 量子比特编号
    - params: 参数字典
    """


    noise = get_single_gate_noise_from_json(qubit, params)
    circuit.append("PAULI_CHANNEL_1", qubit, noise['px_py_pz'])
    circuit.append("DEPOLARIZE1", qubit, noise['p1'])


def add_two_gate_noise(circuit, cx_gate_id, params):
    """
    添加双量子比特门噪声：PAULI_CHANNEL_1（每个量子比特） + DEPOLARIZE2

    Args:
    - circuit: stim.Circuit对象
    - cx_gate_id: CX门ID（字符串）
    - params: 参数字典
    """

    noise = get_two_gate_noise_from_json(cx_gate_id, params)

    # 获取CX门的控制和目标量子比特
    cx_params = params["cx_gates"][cx_gate_id]
    control = cx_params["control"]
    target = cx_params["target"]

    circuit.append("PAULI_CHANNEL_1", control, noise['control_pauli'])
    circuit.append("PAULI_CHANNEL_1", target, noise['target_pauli'])
    circuit.append("DEPOLARIZE2", [control, target], noise['p2'])


def inject_surface_code_noise(base_circuit, data_qubits, x_stabilizers, z_stabilizers,
                              cx_gates, params_file):
    """
    为表面码电路注入噪声通道（不包含泄漏噪声）

    Args:
    - base_circuit: 基础表面码电路
    - data_qubits: 数据比特列表
    - x_stabilizers: X稳定子比特列表
    - z_stabilizers: Z稳定子比特列表
    - cx_gates: CX门映射列表 [(1, (control, target)), ...]
    - params_file: 参数JSON文件路径

    Returns:
    - noisy_circuit: 包含噪声的电路
    """
    # 加载参数
    params = load_surface_code_params(params_file)

    # 创建CX门查找表
    cx_gate_lookup = {}
    for gate_id, (control, target) in cx_gates:
        cx_gate_lookup[(control, target)] = str(gate_id)

    noisy_circuit = stim.Circuit()

    # 处理电路指令
    i = 0
    while i < len(base_circuit):
        instruction = base_circuit[i]

        if instruction.name == "QUBIT_COORDS":
            noisy_circuit.append(instruction.name, instruction.targets_copy(), instruction.gate_args_copy())

        elif instruction.name == "R":
            # R门 + 数据比特初始化噪声
            targets = [t.value for t in instruction.targets_copy()]
            noisy_circuit.append("R", targets)

            # 只对数据比特添加初始化噪声
            add_data_init_noise(noisy_circuit, data_qubits, params)

        elif instruction.name == "H":
            targets = [t.value for t in instruction.targets_copy()]

            for target in targets:
                noisy_circuit.append("H", target)
                add_single_gate_noise(noisy_circuit, target, params)

        elif instruction.name == "CX":
            targets = [t.value for t in instruction.targets_copy()]

            # CX门是成对的
            for j in range(0, len(targets), 2):
                control, target = targets[j], targets[j + 1]

                noisy_circuit.append("CX", [control, target])

                # 查找对应的CX门参数
                gate_key = (control, target)
                if gate_key in cx_gate_lookup:
                    cx_gate_id = cx_gate_lookup[gate_key]
                    add_two_gate_noise(noisy_circuit, cx_gate_id, params)

        elif instruction.name == "MR":
            # MR前：数据比特空闲噪声（仅中间轮） + 稳定子SPAM噪声

            targets = [t.value for t in instruction.targets_copy()]

            stabilizer_qubits = x_stabilizers + z_stabilizers
            measured_stabilizers = [q for q in targets if q in stabilizer_qubits]

            # 检查是否是最后一轮：查看后续是否还有MR指令
            is_last_round = True
            for j in range(i + 1, len(base_circuit)):
                if base_circuit[j].name == "MR":
                    is_last_round = False
                    break

            # 只在非最后轮添加数据比特空闲噪声
            if not is_last_round:
                add_readout_idle_noise(noisy_circuit, data_qubits, params)

            # 稳定子SPAM噪声
            add_measurement_spam_noise(noisy_circuit, measured_stabilizers, params)

            noisy_circuit.append("MR", targets)

        elif instruction.name == "M":
            # M前添加数据比特测量噪声（最终测量，无空闲噪声）
            targets = [t.value for t in instruction.targets_copy()]
            measured_data = [q for q in targets if q in data_qubits]

            add_data_measurement_noise(noisy_circuit, measured_data, params)
            noisy_circuit.append("M", targets)

        else:
            # 其他指令（DETECTOR, OBSERVABLE_INCLUDE等）直接复制
            if hasattr(instruction, 'targets_copy') and hasattr(instruction, 'gate_args_copy'):
                new_targets = []
                for t in instruction.targets_copy():
                    if hasattr(t, 'value'):
                        new_targets.append(t.value if t.value >= 0 else t)
                    else:
                        new_targets.append(t)
                noisy_circuit.append(instruction.name, new_targets, instruction.gate_args_copy())
            else:
                # 对于没有标准方法的指令，尝试直接添加
                try:
                    noisy_circuit += stim.Circuit([instruction])
                except:
                    pass

        i += 1

    return noisy_circuit


