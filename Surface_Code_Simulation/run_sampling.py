import numpy as np
from inject_basic_noise import inject_surface_code_noise
from inject_leakage_noise_vectorized import simulate_surface_code_leakage_vectorized, \
    extract_measurement_affected_vectorized
from surface_code_generate_circuits import generate_surface_code_circuit
from inject_basic_noise import load_surface_code_params
import time


def postprocess_leakage(sampling_results, all_measurement_affected, flip_prob):
    """批量后处理泄漏影响"""
    shots, measurements = sampling_results.shape

    # 预生成随机数
    randoms = np.random.random((shots, measurements))

    # 批量翻转
    flip_mask = (all_measurement_affected == 1) & (randoms < flip_prob)
    results = sampling_results.copy()
    results[flip_mask] = 1 - results[flip_mask]

    return results


def run_sampling(distance, rounds, shots,
                 params_file,
                 flip_prob=0.5, include_leakage=True, verbose=False, basis='z', batch_size=2000):
    """
    运行表面码采样和泄漏后处理

    Args:
        distance: 表面码距离
        rounds: 纠错轮数
        shots: 采样次数
        params_file: 参数文件路径
        flip_prob: 泄漏影响翻转概率
        include_leakage: 是否包含泄漏后处理
        verbose: 是否显示进度
        basis: 'z' 或 'x'，表面码基
        batch_size: 分批处理大小

    Returns:
        {
            'original': 原始采样结果,
            'processed': 泄漏后处理结果,
            'execution_time': 执行时间
        }
    """
    start_time = time.time()

    if verbose:
        print(f"仿真参数: distance={distance}, rounds={rounds}, shots={shots}, flip_prob={flip_prob}, basis={basis}")

    # 加载参数
    if verbose:
        print("加载参数文件...")
    params = load_surface_code_params(params_file)

    # 生成电路和注入噪声
    if verbose:
        print("生成电路和注入噪声...")
    circuit, data_qubits, x_stab, z_stab, cx_gates = generate_surface_code_circuit(distance, rounds, basis)
    noisy_circuit = inject_surface_code_noise(circuit, data_qubits, x_stab, z_stab, cx_gates, params_file)

    # 采样
    if verbose:
        print("开始采样...")
    results = noisy_circuit.compile_sampler().sample(shots=shots)

    if not include_leakage:
        execution_time = time.time() - start_time
        if verbose:
            print(f"总执行时间: {execution_time:.2f}秒")
        return {
            'original': results,
            'processed': results,
            'execution_time': execution_time
        }

    # 泄漏处理
    if verbose:
        print("开始泄漏模拟...")

    # 向量化泄漏模拟
    all_affected_timelines = simulate_surface_code_leakage_vectorized(
        circuit, data_qubits, x_stab, z_stab, cx_gates, params, shots, batch_size
    )

    # 批量提取测量影响状态
    if verbose:
        print("批量提取和后处理...")
    all_affected = extract_measurement_affected_vectorized(
        all_affected_timelines, data_qubits, x_stab, z_stab, rounds
    )

    # 批量后处理
    processed = postprocess_leakage(results, all_affected, flip_prob)

    execution_time = time.time() - start_time

    if verbose:
        print(f"总执行时间: {execution_time:.2f}秒")
        avg_time = execution_time / shots
        print(f"平均每shot: {avg_time:.4f}秒")
        print(f"预计100万shots: {avg_time * 1000000 / 60:.1f}分钟")

    return {
        'original': results,
        'processed': processed,
        'execution_time': execution_time
    }