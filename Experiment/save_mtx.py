import numpy as np
import matplotlib.pyplot as plt
import sys
import os

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from config import NUM_MEA, NUM_ROUNDS, SHOTS_EXP
from anlysis import analyze_job_results

job_ids = []
def compute_correlation_matrix(final_array, num_mea, num_rounds, shots_exp, job_id, save_path):
    """
    Compute and save correlation matrix from measurement data.
    
    Args:
        final_array: numpy array of measurement results_test
        num_mea: number of measurement qubits
        num_rounds: number of measurement rounds
        shots_exp: number of experimental shots
        job_id: job identifier for saving
        save_path: path to save the correlation matrix
    
    Returns:
        correlation_matrix: computed correlation matrix
    """
    # Reshape the data to (shots, rounds, qubits)
    data1 = final_array.reshape(shots_exp, num_rounds, num_mea)
    
    # Transpose data to switch 'rounds' and 'qubits'
    data2 = data1.transpose(0, 2, 1)
    
    # Flatten the data
    flattened_data = data2.reshape(shots_exp, num_mea * num_rounds)
    
    # Compute probabilities
    p_x = np.mean(flattened_data, axis=0)
    p_xi_xj = np.zeros((num_mea * num_rounds, num_mea * num_rounds))
    
    # Compute p_xi_xj
    for k in range(shots_exp):
        temp = np.outer(flattened_data[k], flattened_data[k])
        p_xi_xj += temp
    p_xi_xj /= shots_exp
    
    # Initialize and calculate correlation matrix
    correlation_matrix = np.zeros((num_mea * num_rounds, num_mea * num_rounds))
    xi = p_x.reshape(-1, 1)
    xj = p_x.reshape(1, -1)
    
    term1 = (1 - 2 * xi)
    term2 = (1 - 2 * xj)
    
    valid_indices = np.where((term1 != 0) & (term2 != 0))
    
    for i, j in zip(*valid_indices):
        if i != j:
            correlation_matrix[i, j] = (p_xi_xj[i, j] - xi[i, 0] * xj[0, j]) / (term1[i, 0] * term2[0, j])
    
    # Make matrix symmetric
    np.fill_diagonal(correlation_matrix, 0)
    correlation_matrix = np.triu(correlation_matrix) + np.triu(correlation_matrix, 1).T
    
    # Save the matrix
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, f'correlation_matrix_{job_id}_brisbane_layout_1.npy'), correlation_matrix)
    
    return correlation_matrix

def plot_correlation_matrix(correlation_matrix, num_mea, num_rounds):
    """
    Plot the correlation matrix.
    
    Args:
        correlation_matrix: computed correlation matrix
        num_mea: number of measurement qubits
        num_rounds: number of measurement rounds
    """
    fig, ax = plt.subplots(figsize=(15, 15))
    cax = ax.matshow(correlation_matrix, interpolation='nearest', cmap='plasma', vmin=0, vmax=0.05)
    fig.colorbar(cax)
    
    major_ticks = np.arange(0, num_mea * num_rounds, num_rounds)
    minor_ticks = np.arange(0, num_mea * num_rounds, 1)
    
    major_labels = [f"Q{i+1}" for i in range(num_mea)]
    
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.set_xticklabels(major_labels, minor=False)
    ax.set_yticklabels(major_labels, minor=False)
    
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='minor', linestyle='-', color='gray', linewidth=0.1)
    
    plt.xlabel('Node Index (Qubit-Round Combination)')
    plt.ylabel('Node Index (Qubit-Round Combination)')
    plt.title('Correlation Matrix for Detection Events')
    plt.show()



# Example usage:
if __name__ == "__main__":
    for job_id in job_ids:
        final_array = analyze_job_results(job_id)
        plots_dir = os.path.join(os.path.dirname(__file__), 'data', 'mtx', 'correlation_matrix_brisbane_layout_1')

        matrix = compute_correlation_matrix(
            final_array=final_array,
            num_mea=NUM_MEA,
            num_rounds=NUM_ROUNDS,
            shots_exp=SHOTS_EXP,
            job_id=job_id,
            save_path=plots_dir
        )
    

