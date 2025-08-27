import stim
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)



def generate_surface_code_circuit(distance, rounds, basis='z'):
    """
    生成基于CX门的表面码电路，量子比特从1开始连续编号

    Args:
    - distance: 表面码距离
    - rounds: 纠错轮数
    - basis: 'z' 或 'x'，指定Z基或X基表面码

    Returns:
    - circuit: 转换后的电路
    - data_qubits: 数据比特序号列表
    - x_stabilizers: X稳定子比特序号列表
    - z_stabilizers: Z稳定子比特序号列表
    - cx_gates: CX门的映射列表，格式为[(1, (control, target)), (2, (control, target)), ...]
    """
    # 根据basis选择表面码类型
    if basis.lower() == 'z':
        code_type = "surface_code:rotated_memory_z"
    elif basis.lower() == 'x':
        code_type = "surface_code:rotated_memory_x"
    else:
        raise ValueError("basis must be 'z' or 'x'")

    # 生成原始电路
    base_circuit = stim.Circuit.generated(
        code_type,
        distance=distance,
        rounds=rounds,
    )

    # 解析量子比特坐标和重新编号
    qubit_coords = {}
    original_to_new = {}
    new_id = 1

    for instruction in base_circuit:
        if instruction.name == "QUBIT_COORDS":
            x, y = instruction.gate_args_copy()
            original_id = instruction.targets_copy()[0].value
            qubit_coords[new_id] = (x, y)
            original_to_new[original_id] = new_id
            new_id += 1

    # 分类量子比特
    data_qubits = []
    x_stabilizers = []
    z_stabilizers = []

    # 找到最后的测量指令获取数据比特
    for instruction in reversed(base_circuit):
        if instruction.name in ["M", "MX"]:
            data_qubits = [original_to_new[t.value] for t in instruction.targets_copy()]
            break

    # 找到第一个H门指令获取X稳定子
    for instruction in base_circuit:
        if instruction.name == "H":
            x_stabilizers = [original_to_new[t.value] for t in instruction.targets_copy()]
            break

    # 其余稳定子比特为Z稳定子
    all_qubits = set(qubit_coords.keys())
    z_stabilizers = list(all_qubits - set(data_qubits) - set(x_stabilizers))

    # 构建新电路并提取CX门
    new_circuit = stim.Circuit()
    cx_gates = []
    cx_index = 1  # CX门编号从1开始

    # 添加量子比特坐标
    for new_id, (x, y) in qubit_coords.items():
        new_circuit.append("QUBIT_COORDS", [new_id], [x, y])

    def process_cx_instruction(instruction):
        """处理CX指令并提取门信息"""
        nonlocal cx_index
        targets = instruction.targets_copy()
        new_targets = []

        # CX门是成对出现的，每两个target为一对
        for j in range(0, len(targets), 2):
            control = original_to_new[targets[j].value]
            target = original_to_new[targets[j + 1].value]

            # 检查这个CX门是否已经存在
            gate_key = (control, target)
            existing_gate_id = None
            for gate_id, (ctrl, tgt) in cx_gates:
                if ctrl == control and tgt == target:
                    existing_gate_id = gate_id
                    break

            if existing_gate_id is None:
                # 新的CX门，分配新ID
                cx_gates.append((cx_index, (control, target)))
                cx_index += 1
            else:
                # 重复的CX门，重用已有ID
                cx_gates.append((existing_gate_id, (control, target)))

            new_targets.extend([control, target])

        return new_targets

    def convert_targets(instruction):
        """转换指令的target"""
        if not instruction.targets_copy():
            return []

        new_targets = []
        for t in instruction.targets_copy():
            if t.value >= 0 and t.value in original_to_new:
                new_targets.append(original_to_new[t.value])
            else:
                new_targets.append(t)
        return new_targets

    # 转换指令
    i = 0
    while i < len(base_circuit):
        instruction = base_circuit[i]

        if instruction.name == "QUBIT_COORDS":
            i += 1
            continue

        elif instruction.name == "CX":
            new_targets = process_cx_instruction(instruction)
            new_circuit.append("CX", new_targets)


        elif instruction.name == "RX":

            # 收集RX的targets

            rx_targets = convert_targets(instruction)

            # 查找后续的R指令并合并

            all_r_targets = rx_targets.copy()

            j = i + 1

            while j < len(base_circuit) and base_circuit[j].name in ["TICK", "R"]:

                if base_circuit[j].name == "R":
                    r_targets = convert_targets(base_circuit[j])

                    all_r_targets.extend(r_targets)

                    # 跳过这个R指令

                    i = j  # 让外层循环跳过这个R指令

                    break

                j += 1

            # 添加合并后的R门和H门

            new_circuit.append("R", all_r_targets)

            new_circuit.append("H", rx_targets)

        elif instruction.name == "MX":
            # 将MX转换为H+M
            new_targets = convert_targets(instruction)
            new_circuit.append("H", new_targets)
            new_circuit.append("M", new_targets, instruction.gate_args_copy())

        elif instruction.name == "REPEAT":
            # 处理REPEAT块
            repeat_count = instruction.repeat_count
            repeat_body = instruction.body_copy()

            # 处理REPEAT体内的指令
            new_repeat_body = stim.Circuit()
            for sub_instruction in repeat_body:
                if sub_instruction.name == "CX":
                    sub_new_targets = process_cx_instruction(sub_instruction)
                    new_repeat_body.append("CX", sub_new_targets)
                else:
                    sub_new_targets = convert_targets(sub_instruction)
                    if sub_new_targets:
                        new_repeat_body.append(sub_instruction.name, sub_new_targets, sub_instruction.gate_args_copy())
                    else:
                        new_repeat_body.append(sub_instruction.name, [], sub_instruction.gate_args_copy())

            # 添加新的REPEAT块 - 手动展开重复内容
            for _ in range(repeat_count):
                new_circuit += new_repeat_body

        else:
            # 其他指令
            new_targets = convert_targets(instruction)
            if new_targets:
                new_circuit.append(instruction.name, new_targets, instruction.gate_args_copy())
            else:
                new_circuit.append(instruction.name, [], instruction.gate_args_copy())

        i += 1

    return new_circuit, data_qubits, x_stabilizers, z_stabilizers, cx_gates
'''
new_circuit, data_qubits, x_stabilizers, z_stabilizers, cx_gates = generate_surface_code_circuit(3, 3, basis='x')

print(new_circuit)
print(data_qubits)
print(x_stabilizers)
print(z_stabilizers)
print(cx_gates)'''





