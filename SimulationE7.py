import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
R = 10  # Number of cysteine residues
num_molecules = 10000
num_steps = 60
initial_k10 = 1650
initial_k0 = 8350

# Generate all i-states as binary strings
def generate_i_states(r):
    num_states = 2**r
    i_states = [bin(i)[2:].zfill(r) for i in range(num_states)]
    return i_states

# Initialize transition probabilities
def initialize_transitions(i_states, r):
    transition_dict = {}
    for state in i_states:
        current_k = state.count('1')
        allowed = {}
        for i in range(r):
            new_state = list(state)
            new_state[i] = '1' if new_state[i] == '0' else '0'
            new_state = ''.join(new_state)
            new_k = new_state.count('1')
            if abs(new_k - current_k) == 1:
                allowed[new_state] = 0.01 if new_k > current_k else 0.75 if new_k < current_k else 0.24
        allowed[state] = 1.0 - sum(allowed.values())  # Remaining probability for staying
        transition_dict[state] = allowed
    return transition_dict

# Normalize transition probabilities
def normalize_transitions(transition_dict):
    for state, transitions in transition_dict.items():
        total_prob = sum(transitions.values())
        if total_prob != 1.0:
            for next_state in transitions:
                transitions[next_state] /= total_prob
    return transition_dict

# Initialize population
def initialize_population(i_states, r):
    population = {state: 0 for state in i_states}
    for state in i_states:
        if state.count('1') == 10:
            population[state] = initial_k10 // (len(i_states) // (r + 1))
        elif state.count('1') == 0:
            population[state] = initial_k0 // (len(i_states) // (r + 1))
    return population

# Calculate Ricci curvature
def calculate_ricci_curvature(k, r):
    if k <= 2:
        return 1 - 0.1 * k
    elif k >= r - 2:
        return 1 - 0.1 * (r - k)
    else:
        return 0.5  # Flat region in the middle

# Shannon entropy calculation
def calculate_shannon_entropy(population, total_population):
    probabilities = np.array([count / total_population for count in population.values() if count > 0])
    return -np.sum(probabilities * np.log(probabilities))

# Lyapunov exponent calculation
def calculate_lyapunov_exponent(history):
    differences = []
    for t in range(1, len(history)):
        prev_dist = np.array([v for v in history[t-1][0].values()])  # Extract dictionary from tuple
        curr_dist = np.array([v for v in history[t][0].values()])    # Extract dictionary from tuple
        divergence = np.abs(curr_dist - prev_dist)
        differences.append(np.mean(np.log(divergence + 1e-9)))  # Avoid log(0) by adding a small value
    return np.mean(differences)


# Run simulation
def run_simulation(transition_dict, population, num_steps, r):
    total_population = sum(population.values())
    history = []

    for step in range(num_steps):
        new_population = {state: 0 for state in population}

        for state, count in population.items():
            for next_state, prob in transition_dict[state].items():
                new_population[next_state] += count * prob

        # Normalize population
        total_new_population = sum(new_population.values())
        scale_factor = total_population / total_new_population
        new_population = {state: round(count * scale_factor) for state, count in new_population.items()}

        # Record history
        entropy = calculate_shannon_entropy(new_population, total_population)
        history.append((new_population.copy(), entropy))

        population = new_population

    return history

# Visualize 3D k-space manifold
def plot_k_space_3d(population, r):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = [], [], []
    sizes = []

    for state, count in population.items():
        k = state.count('1')
        x.append(k)
        y.append(count)
        z.append(calculate_ricci_curvature(k, r))
        sizes.append(count / 10)

    ax.scatter(x, y, z, s=sizes, c=z, cmap='viridis')
    ax.set_xlabel('k-State')
    ax.set_ylabel('Population')
    ax.set_zlabel('Ricci Curvature')
    plt.title('3D k-Space Manifold with Ricci Curvature')
    plt.savefig('3D_k_space_manifold.png', dpi=300)
    plt.show()

# Main Execution
i_states = generate_i_states(R)
transitions = initialize_transitions(i_states, R)
transitions = normalize_transitions(transitions)
population = initialize_population(i_states, R)
history = run_simulation(transitions, population, num_steps, R)

# Final Outputs
final_population = history[-1][0]
final_entropy = history[-1][1]
lyapunov_exponent = calculate_lyapunov_exponent(history)

print(f"Final Population Count: {sum(final_population.values())}")
print(f"Final Shannon Entropy: {final_entropy:.4f}")
print(f"Lyapunov Exponent: {lyapunov_exponent:.4f}")

# Visualize Final Results
plot_k_space_3d(final_population, R)

# Export trajectory as CSV for further analysis
import pandas as pd
df = pd.DataFrame(trajectory_matrix, columns=[f"k={k}" for k in range(R + 1)])
df.to_csv("trajectory.csv", index=False)
