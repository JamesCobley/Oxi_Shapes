# Install necessary libraries
!pip install numpy matplotlib seaborn

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Define parameters
R = 10  # Number of cysteine residues
num_steps = 60  # Number of simulation steps
initial_population = 10000  # Initial population in k=10
proteoforms = [bin(i)[2:].zfill(R) for i in range(2**R)]  # Binary redox strings

# Transition probabilities (P terms)
P_oxidation = {k: 0.01 if k < R else 0.0 for k in range(R + 1)}  # Oxidation probabilities
P_reduction = {k: 0.0 if k == 0 else 0.15 if k == 10 else 0.75 if k == 9 else 0.0 for k in range(R + 1)}  # Reduction probabilities
P_stay = {k: 1.0 - (P_oxidation[k] + P_reduction[k]) for k in range(R + 1)}  # Probability of remaining in the same state

# Ricci curvature function
def ricci_curvature(from_k, to_k, direction):
    if direction == "oxidizing":
        if 0 <= from_k <= 2:
            return 0.5 + (0.5 * (from_k / 2))  # Steep gradient for k = 0 to 2
        return 1.0  # Constant for k > 2
    elif direction == "reducing":
        if 9 <= from_k <= 10:
            return 0.5 + (0.5 * ((10 - from_k) / 1))  # Steep gradient for k = 10 to 9
        return 0.5  # Shallower for k < 9

# Energy function
def energy(from_k, to_k, direction):
    if direction == "oxidizing":
        return 5 + (5 * (from_k / R))  # Scales energy input for oxidizing
    elif direction == "reducing":
        return 5 + (5 * ((R - from_k) / R))  # Scales energy input for reducing

# Initialize population: dictionary of binary states with populations
population = defaultdict(int)
population["1" * R] = 1650  # Start with 1,650 molecules in k = 10 (1111111111)
population["0" * R] = 8350  # Start with 8,350 molecules in k = 0 (0000000000)


# Find allowed transitions for each proteoform
def find_allowed_transitions(proteoform):
    struct_vec = list(proteoform)
    current_k = struct_vec.count('1')  # Current k-value
    allowed_transitions = {}
    for i in range(R):
        new_struct = struct_vec.copy()
        new_struct[i] = '1' if new_struct[i] == '0' else '0'  # Toggle site
        new_proteoform = ''.join(new_struct)
        new_k = new_struct.count('1')
        if abs(new_k - current_k) == 1:  # Stepwise move
            allowed_transitions[new_proteoform] = 1  # Allowed transition
    return allowed_transitions

# Transition matrix based on allowed transitions
transition_matrices = {}
for proteoform in proteoforms:
    allowed_transitions = find_allowed_transitions(proteoform)
    transition_matrices[proteoform] = allowed_transitions

# Simulate movement across k-space
trajectory = []
for step in range(num_steps):
    new_population = defaultdict(int)
    for proteoform, count in population.items():
        current_k = proteoform.count('1')  # Current k-state
        transitions = transition_matrices[proteoform]
        for next_proteoform in proteoforms:
            next_k = next_proteoform.count('1')
            direction = "oxidizing" if next_k > current_k else "reducing"
            curvature = ricci_curvature(current_k, next_k, direction)
            energy_input = energy(current_k, next_k, direction)
            if next_proteoform in transitions:  # Allowed transition
                transition_prob = P_oxidation[current_k] if direction == "oxidizing" else P_reduction[current_k]
                new_population[next_proteoform] += count * transition_prob * energy_input * curvature
            elif next_proteoform == proteoform:  # Stay transition
                new_population[next_proteoform] += count * P_stay[current_k]
    population = new_population
    trajectory.append(population.copy())

# Visualization: Population trajectory in k-space
trajectory_matrix = np.zeros((num_steps, R + 1))
for step, pop in enumerate(trajectory):
    for proteoform, count in pop.items():
        k_value = proteoform.count('1')
        trajectory_matrix[step, k_value] += count

plt.figure(figsize=(12, 6))
sns.heatmap(trajectory_matrix.T, cmap="viridis", xticklabels=10, yticklabels=1)
plt.title("Population Trajectory Across k-space")
plt.xlabel("Time Steps")
plt.ylabel("k-State")
plt.show()

# Visualization: Ricci curvature in k-space
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
k_from = np.arange(R + 1)
k_to = np.arange(R + 1)
X, Y = np.meshgrid(k_from, k_to)
Z = np.array([[ricci_curvature(kf, kt, "oxidizing" if kt > kf else "reducing") 
               for kt in k_to] for kf in k_from])
ax.plot_surface(X, Y, Z, cmap="coolwarm")
plt.title("Ricci Curvature Across k-space")
plt.show()

# Export trajectory as CSV for further analysis
import pandas as pd
df = pd.DataFrame(trajectory_matrix, columns=[f"k={k}" for k in range(R + 1)])
df.to_csv("trajectory.csv", index=False)
