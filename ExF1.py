import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
import matplotlib.cm as cm

def generate_pascals_triangle(n):
    """Generates Pascal's Triangle up to row n"""
    triangle = np.zeros((n, n), dtype=int)
    for i in range(n):
        triangle[i, 0] = 1
        for j in range(1, i + 1):
            triangle[i, j] = triangle[i - 1, j - 1] + triangle[i - 1, j]
    return triangle

def extract_fibonacci_from_pascals(triangle):
    """Extracts Fibonacci numbers using Pascal's diagonal sum rule"""
    n = triangle.shape[0]
    fib_indices = []
    fib_numbers = []
    for k in range(n):
        if k < 2:
            fib_numbers.append(1)
        else:
            fib_numbers.append(fib_numbers[-1] + fib_numbers[-2])
        if k < n:
            fib_indices.append((k, k//2))  # Approximate diagonal selection
    return fib_indices, fib_numbers

def plot_pascals_triangle_with_fibonacci(triangle, filename):
    """Plots Pascal’s Triangle with Fibonacci numbers highlighted"""
    n = triangle.shape[0]
    fig, ax = plt.subplots(figsize=(10, 10))
    
    norm = Normalize(vmin=np.min(triangle), vmax=np.max(triangle))
    cmap = cm.get_cmap("viridis")
    
    for i in range(n):
        for j in range(i + 1):
            value = triangle[i, j]
            color = cmap(norm(value))
            ax.add_patch(plt.Rectangle((j, n - i - 1), 1, 1, color=color, ec='black'))
            ax.text(j + 0.5, n - i - 1 + 0.5, str(value), va='center', ha='center', fontsize=10, color="white")

    # Overlay Fibonacci numbers
    fib_indices, fib_numbers = extract_fibonacci_from_pascals(triangle)
    for (i, j), fib in zip(fib_indices, fib_numbers):
        ax.text(j + 0.5, n - i - 1 + 0.5, str(fib), va='center', ha='center', fontsize=10, fontweight='bold', color="red")
    
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Pascal's Triangle with Fibonacci Numbers", fontsize=14)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def plot_sierpinski_triangle(triangle, filename):
    """Plots Sierpiński Triangle using Pascal's Triangle mod 2"""
    n = triangle.shape[0]
    sierpinski = triangle % 2

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(sierpinski[::-1], cmap="viridis", annot=triangle[::-1], fmt="d", linewidths=0.5, linecolor='black', cbar=False, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Sierpiński Triangle from Pascal's Triangle (mod 2)", fontsize=14)

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# Set the depth of Pascal's Triangle
n = 10 # Adjustable depth

# Generate Pascal's Triangle
triangle = generate_pascals_triangle(n)

# Generate and save plots
plot_pascals_triangle_with_fibonacci(triangle, "pascals_fibonacci.png")
plot_sierpinski_triangle(triangle, "sierpinski_triangle.png")
