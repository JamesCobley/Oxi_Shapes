import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Define R range from 1 to 100
R = np.arange(1, 50)

# Compute i-space cardinality using binomial explosion: |H_i| = 2^R
i_space_cardinality = 2**R

# Set up figure with viridis colormap
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# Normalize color mapping
norm = mcolors.Normalize(vmin=min(R), vmax=max(R))
cmap = plt.cm.viridis

# Scatter plot with viridis gradient
scatter = ax.scatter(R, i_space_cardinality, c=R, cmap=cmap, norm=norm, edgecolor='k', alpha=0.8)

# Ensure x-axis fully includes 1 to 100
ax.set_xlim(0, 50)

# Log scale for y-axis
ax.set_yscale("log")

# Labels and title
ax.set_xlabel("R", fontsize=12)
ax.set_ylabel("i-Space Cardinality", fontsize=12)


# Colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("R", fontsize=12)

# Grid and styling
ax.grid(True, which="both", linestyle="--", alpha=0.5)

# Save plot as PNG in the /content/ directory
save_path = "/content/i_space_log_plot_fixed.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")

# Show plot
plt.show()

# Print file path for easy download
print(f"Plot saved as: {save_path}")
