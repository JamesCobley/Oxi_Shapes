import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

# Function to generate binary proteoforms for given R
def generate_proteoforms(R):
    return [format(i, f'0{R}b') for i in range(2**R)]

# Function to create an adjacency matrix for given R
def create_adjacency_matrix(proteoforms):
    n = len(proteoforms)
    adj_matrix = np.zeros((n, n), dtype=int)
    
    for i, s1 in enumerate(proteoforms):
        for j, s2 in enumerate(proteoforms):
            if sum(c1 != c2 for c1, c2 in zip(s1, s2)) == 1:  # Single-bit flip
                adj_matrix[i, j] = 1
    return adj_matrix

# Function to plot and save adjacency matrix
def plot_and_save_adjacency_matrix(R):
    proteoforms = generate_proteoforms(R)
    adj_matrix = create_adjacency_matrix(proteoforms)

    plt.figure(figsize=(8, 6))
    sns.heatmap(adj_matrix, annot=True, cbar=True, square=True, 
                xticklabels=proteoforms, yticklabels=proteoforms, 
                cmap="viridis", linecolor='black', linewidths=0.5)
    
    plt.title(f"Adjacency Matrix for R = {R}")
    plt.xlabel("Proteoform State")
    plt.ylabel("Proteoform State")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    filename = f"adjacency_matrix_R{R}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

# Generate and save adjacency matrices for R = 2, 4, 5
for R in [2, 4, 5]:
    plot_and_save_adjacency_matrix(R)

# Display completion message
print("All adjacency matrix images are saved as PNG files with 300 DPI.")
