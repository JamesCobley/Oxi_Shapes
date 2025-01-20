!pip install networkx
!pip install GraphRicciCurvature
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from mpl_toolkits.mplot3d import Axes3D

# Initialize parameters
R = 10  # Number of cysteines
num_steps = 60  # Number of simulation steps
initial_population = {'0000000000': 8350, '1111111111': 1650}

# Define k-manifold P-matrix
k_manifold_P = {
    0: {"oxidation": 0.01, "reduction": 0.0, "stay": 0.99},
    1: {"oxidation": 0.01, "reduction": 0.9, "stay": 0.09},
    2: {"oxidation": 0.02, "reduction": 0.96, "stay": 0.05},
    3: {"oxidation": 0.03, "reduction": 0.92, "stay": 0.05},
    4: {"oxidation": 0.04, "reduction": 0.91, "stay": 0.05},
    5: {"oxidation": 0.05, "reduction": 0.90, "stay": 0.05},
    6: {"oxidation": 0.06, "reduction": 0.89, "stay": 0.05},
    7: {"oxidation": 0.07, "reduction": 0.88, "stay": 0.05},
    8: {"oxidation": 0.08, "reduction": 0.87, "stay": 0.05},
    9: {"oxidation": 0.25, "reduction": 0.70, "stay": 0.05},
    10: {"oxidation": 0.0, "reduction": 0.001, "stay": 0.999},
}

# Ricci curvature parameters
ricci_curvature = {k: 0.8 if k <= 2 else (0.95 if k >= 9 else 1.0) for k in range(R + 1)}

# Generate all binary i-states
def generate_proteoforms(r):
    num_states = 2**r
    proteoforms = [format(i, f'0{r}b') for i in range(num_states)]
    return proteoforms

# Generate transition matrix using the k-manifold P-matrix with normalization
def generate_transition_matrix(r, k_manifold_P):
    proteoforms = generate_proteoforms(r)
    transition_matrix = {}
    for proteoform in proteoforms:
        current_k = proteoform.count('1')
        allowed = {}

        # Apply k-manifold P-matrix
        allowed[proteoform] = k_manifold_P[current_k]["stay"]
        for i in range(r):
            toggled = list(proteoform)
            toggled[i] = '1' if toggled[i] == '0' else '0'
            toggled = ''.join(toggled)
            new_k = toggled.count('1')

            if new_k == current_k + 1:  # Oxidation
                allowed[toggled] = k_manifold_P[current_k]["oxidation"]
            elif new_k == current_k - 1:  # Reduction
                allowed[toggled] = k_manifold_P[current_k]["reduction"]

        # Normalize probabilities to ensure they sum to 1
        total_prob = sum(allowed.values())
        if total_prob > 0:  # Avoid division by zero
            allowed = {state: prob / total_prob for state, prob in allowed.items()}

        barred = [p for p in proteoforms if p not in allowed]
        transition_matrix[proteoform] = (allowed, barred)

    return transition_matrix

# Update population with conservation check
def update_population(population, transition_matrix):
    new_population = defaultdict(float)
    for state, count in population.items():
        allowed_transitions, _ = transition_matrix[state]
        for target_state, prob in allowed_transitions.items():
            new_population[target_state] += count * prob

    # Verify population conservation
    total_population = sum(new_population.values())
    print(f"Population Conservation: {sum(population.values())} -> {total_population}")
    return new_population

# Ensure all states are represented in distributions
def fill_population_with_zeros(population, all_states):
    return {state: population.get(state, 0) for state in all_states}

# Shannon entropy calculation
def calculate_shannon_entropy(population, total_population):
    probs = np.array([count / total_population for count in population.values() if count > 0])
    return -np.sum(probs * np.log2(probs))

# Lyapunov exponent calculation
def calculate_lyapunov_exponent(history, all_states):
    differences = []
    for t in range(1, len(history)):
        prev_dist = np.array([history[t - 1][0].get(state, 0) for state in all_states])
        curr_dist = np.array([history[t][0].get(state, 0) for state in all_states])
        divergence = np.abs(curr_dist - prev_dist)
        differences.append(np.mean(np.log(divergence + 1e-9)))  # Avoid log(0)
    return np.mean(differences)

# Initialize simulation
proteoforms = generate_proteoforms(R)
transition_matrix = generate_transition_matrix(R, k_manifold_P)
population = initial_population.copy()
total_population = sum(initial_population.values())
history = []

# Run simulation
for step in range(num_steps):
    filled_population = fill_population_with_zeros(population, proteoforms)
    entropy = calculate_shannon_entropy(filled_population, total_population)
    history.append((filled_population, entropy))
    print(f"Step {step}: Shannon Entropy = {entropy:.4f}")
    population = update_population(population, transition_matrix)

# Final calculations
filled_final_population = fill_population_with_zeros(history[-1][0], proteoforms)
final_entropy = history[-1][1]
lyapunov_exponent = calculate_lyapunov_exponent(history, proteoforms)

# Compute mean redox state
mean_redox_state = sum(k * count for k, count in [(state.count('1'), count) for state, count in filled_final_population.items()]) / total_population

print(f"Final Population Count: {sum(filled_final_population.values()):.0f}")
print(f"Final Shannon Entropy: {final_entropy:.4f}")
print(f"Lyapunov Exponent: {lyapunov_exponent:.4f}")
print(f"Mean Redox State of cdc20 Population: {mean_redox_state:.4f}")

# Compute k-manifold Ricci curvature
kspace_graph_by_k = nx.Graph()
k_manifold_population = defaultdict(float)

# Aggregate k-manifold populations
for state, count in filled_final_population.items():
    k_value = state.count('1')
    k_manifold_population[k_value] += count
    kspace_graph_by_k.add_node(k_value)

# Connect k-manifolds in the graph
for k1 in k_manifold_population.keys():
    for k2 in k_manifold_population.keys():
        if abs(k1 - k2) == 1:  # Connect adjacent k-states
            kspace_graph_by_k.add_edge(k1, k2)

# Compute Ricci curvature for k-manifolds
orc_k = OllivierRicci(kspace_graph_by_k, alpha=0.5)
orc_k.compute_ricci_curvature()
k_curvatures = nx.get_edge_attributes(orc_k.G, "ricciCurvature")

# Visualize Ricci curvature for k-manifolds
plt.figure(figsize=(12, 10))
edge_colors = [k_curvatures.get(edge, 0) for edge in kspace_graph_by_k.edges()]
pos_k = nx.spring_layout(kspace_graph_by_k)
nx.draw(
    kspace_graph_by_k, pos_k, with_labels=True,
    edge_color=edge_colors, edge_cmap=plt.cm.coolwarm,
    node_size=500, font_size=10
)
sm = plt.cm.ScalarMappable(
    cmap=plt.cm.coolwarm,
    norm=plt.Normalize(vmin=min(edge_colors), vmax=max(edge_colors))
)
sm.set_array([])

plt.title("Ricci Curvature on k-Manifold Graph")
plt.savefig("/content/ricci.png", dpi=300)
plt.show()

# Visualize final 2D k-space basins
plt.figure(figsize=(10, 6))
sns.barplot(x=list(k_manifold_population.keys()), y=list(k_manifold_population.values()), palette="viridis")
plt.title("Final Basin Sizes in k-Space")
plt.xlabel("k-State")
plt.ylabel("Population Size")
plt.grid()
plt.savefig("/content/k_space_basins.png", dpi=300)
plt.show()

# 3D Visualization of k-Space Dynamics from Simulation Results
k_states = np.arange(0, R + 1)
population_sizes = [k_manifold_population.get(k, 0) for k in k_states]
ricci_curvature_values = [k_curvatures.get((k, k+1), 0) if k < R else k_curvatures.get((k-1, k), 0) for k in k_states]

# Create a grid for the surface
X, Y = np.meshgrid(k_states, population_sizes)
Z = np.tile(ricci_curvature_values, (len(population_sizes), 1))

# Plot 3D Surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# Add titles and labels
ax.set_title("3D Visualization of k-Space Dynamics (From Results)", fontsize=14)
ax.set_xlabel("k-State (Oxidation Level)")
ax.set_ylabel("Population Size")
ax.set_zlabel("Ricci Curvature")
plt.colorbar(surf, shrink=0.5, aspect=5, label="Ricci Curvature")
plt.savefig("/content/simulation_based_E7_geo.png", dpi=300)
plt.show()
