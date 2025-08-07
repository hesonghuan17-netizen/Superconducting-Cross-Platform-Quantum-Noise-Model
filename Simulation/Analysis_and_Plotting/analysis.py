import numpy as np


def compute_correlation_matrix(data, num_qubits, rounds_, shots):
    """
    Computes the correlation matrix from quantum measurement data.

    Parameters:
    - data : np.array
        The results_test array from quantum measurements.
    - num_qubits : int
        Total number of qubits.
    - rounds : int
        Number of measurement rounds.
    - shots : int
        Total number of measurement shots.

    Returns:
    - np.array
        The computed correlation matrix.
    """
    num_mea = (num_qubits - 1) // 2
    # Reshape the data to (shots, rounds, qubits)
    data1 = data.reshape(shots, rounds_, num_mea)

    # Transpose data to switch 'rounds' and 'qubits' to make qubits sequential per shot
    data2 = data1.transpose(0, 2, 1)

    # Flatten the data back to (shots, qubits * rounds)
    flattened_data = data2.reshape(shots, num_mea * rounds_)

    # Compute probabilities
    p_x = np.mean(flattened_data, axis=0)
    p_xi_xj = np.zeros((num_mea * rounds_, num_mea * rounds_))

    # Efficient computation of p_xi_xj using vectorized operations
    for k in range(shots):
        temp = np.outer(flattened_data[k], flattened_data[k])
        p_xi_xj += temp

    p_xi_xj /= shots

    # Initialize correlation matrix
    correlation_matrix = np.zeros((num_mea * rounds_, num_mea * rounds_))

    # Calculate correlations efficiently
    xi = p_x.reshape(-1, 1)
    xj = p_x.reshape(1, -1)

    xi_xj = p_xi_xj

    term1 = (1 - 2 * xi)
    term2 = (1 - 2 * xj)

    # Avoid division by zero by checking where terms are non-zero
    valid_indices = np.where((term1 != 0) & (term2 != 0))

    # Compute correlation matrix for valid indices using correct array indexing
    for i, j in zip(*valid_indices):
        if i != j:
            correlation_matrix[i, j] = (xi_xj[i, j] - xi[i, 0] * xj[0, j]) / (term1[i, 0] * term2[0, j])

    # Since the correlation matrix is symmetric, mirror the results_test
    np.fill_diagonal(correlation_matrix, 0)
    correlation_matrix_sim = np.triu(correlation_matrix) + np.triu(correlation_matrix, 1).T

    return correlation_matrix_sim

