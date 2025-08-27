import json
import random
from surface_code_generate_circuits import generate_surface_code_circuit



def generate_surface_code_json(circuit, data_qubits, x_stabilizers, z_stabilizers, cx_gates,
                               output_file="surface_code_params.json"):
    """
    根据表面码电路信息生成对应格式的JSON参数文件

    Args:
    - circuit: 表面码电路
    - data_qubits: 数据比特列表
    - x_stabilizers: X稳定子比特列表
    - z_stabilizers: Z稳定子比特列表
    - cx_gates: CX门映射列表 [(1, (control, target)), ...]
    - output_file: 输出文件名

    Returns:
    - params_dict: 生成的参数字典
    """

    # 获取所有量子比特
    all_qubits = sorted(set(data_qubits + x_stabilizers + z_stabilizers))

    # 生成量子比特参数
    qubits_params = {}

    for qubit in all_qubits:
        # 为每个量子比特生成合理的随机参数
        qubits_params[str(qubit)] = {
            "measurement_spam_rate": round(random.uniform(0.01, 0.05), 6),
            "data_init_error": round(random.uniform(0.001, 0.003), 6),
            "data_measurement_error": round(random.uniform(0.002, 0.004), 6),
            "t1": round(random.uniform(0.0001, 0.0006), 10),
            "t2": round(random.uniform(0.00005, 0.0005), 10),
            "sqg_fid": round(random.uniform(0.999, 0.9999), 10),
            "sqg_length": 6e-08,  # 固定值
            "rd_length": 1.3e-06,  # 固定值
            "lp": round(random.uniform(0.0001, 0.002), 10),
            "sp": round(random.uniform(0, 0.01), 6)
        }

    # 生成CX门参数
    cx_gates_params = {}

    for gate_idx, (control, target) in cx_gates:
        cx_gates_params[str(gate_idx)] = {
            "control": control,
            "target": target,
            "cx_fid": round(random.uniform(0.98, 0.998), 6),
            "cx_length": round(random.uniform(6.0e-07, 7.5e-07), 10),
            "lp_propagation_prob": round(random.uniform(0, 0.1), 6)
        }

    # 构建完整的参数字典
    params_dict = {
        "qubits": qubits_params,
        "cx_gates": cx_gates_params,
        "_metadata": {
            "total_qubits": len(all_qubits),
            "data_qubits": data_qubits,
            "x_stabilizers": x_stabilizers,
            "z_stabilizers": z_stabilizers,
            "total_cx_gates": len(cx_gates),
            "created_by": "generate_surface_code_json",
            "description": "Auto-generated surface code parameters with random values for testing"
        }
    }

    # 保存到JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(params_dict, f, indent=2, ensure_ascii=False)



    return params_dict


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



