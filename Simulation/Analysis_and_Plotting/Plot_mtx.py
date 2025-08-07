import numpy as np
import matplotlib.pyplot as plt
import sys
import os
#from objective_function_leakage import M_sim



root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#print(root_path)
sys.path.append(root_path)
correlation_matrix_sim = np.load(
    r'')


from config import NUM_QUBITS, NUM_ROUNDS, SHOTS, SHOTS_EXP

num_mea = (NUM_QUBITS - 1) // 2
shots = SHOTS*SHOTS_EXP

#shots, qubits, rounds_ = shots, num_mea, (rounds+2)
shots, qubits, rounds_ = shots, num_mea, NUM_ROUNDS

fig, ax = plt.subplots(figsize=(15, 15))
cax = ax.matshow(correlation_matrix_sim, interpolation='nearest', cmap='plasma',vmin=0, vmax=0.05)
fig.colorbar(cax)

# Set major and minor ticks to properly match the axes
# Major ticks at the start of each qubit's rounds
major_ticks = np.arange(0, qubits * rounds_, rounds_)
minor_ticks = np.arange(0, qubits * rounds_, 1)

# Set labels for major ticks (Rounds)
minor_labels = [f"R{j+1}" for j in range(rounds_)]  # Labels for qubits
major_labels = [f"Q{i+1}" for i in range(qubits)]  # Labels for rounds

ax.set_xticks(major_ticks)
ax.set_yticks(major_ticks)
ax.set_xticklabels(major_labels, minor=False)
ax.set_yticklabels(major_labels, minor=False)

# Set grid for minor ticks
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(minor_ticks, minor=True)
ax.grid(which='minor', linestyle='-', color='gray', linewidth=0.1)

plt.xlabel('Node Index (Qubit-Round Combination)')
plt.ylabel('Node Index (Qubit-Round Combination)')
plt.title('Correlation Matrix for Detection Events')
plt.show()
