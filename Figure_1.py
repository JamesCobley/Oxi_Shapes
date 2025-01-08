import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Function to generate Pascal's Triangle
def generate_pascals_triangle(rows):
    triangle = [[1]]
    for i in range(1, rows):
        row = [1]
        for j in range(1, i):
            row.append(triangle[i-1][j-1] + triangle[i-1][j])
        row.append(1)
        triangle.append(row)
    return triangle

# Function to create the convoluted wave
def create_convoluted_wave(k_values, contributions, x, sigma=0.1):
    return sum(contribution * np.exp(-((x - k) ** 2) / (2 * sigma ** 2)) for k, contribution in zip(k_values, contributions))

# Data for the table
table_data = [
    ["k", "%-oxidised", "i-states", "Structure"],
    ["0", "0", "1", "[000]"],
    ["1", "33.3", "3", "[100], [010], [001]"],
    ["2", "66.6", "3", "[110], [011], [101]"],
    ["3", "100", "1", "[111]"],
]

# Generate Pascal's Triangle
rows = 11
triangle = generate_pascals_triangle(rows)

# Define k-space values and their contributions
k_values = np.array([0, 1, 2, 3])
contributions = np.array([0.12, 0.38, 0.38, 0.12])

# Generate x values for the wave
x = np.linspace(-1, 4, 500)

# Create the convoluted wave
convoluted_wave = create_convoluted_wave(k_values, contributions, x)

# Create the bar chart data (deconvoluted components)
deconvoluted_contributions = contributions * 100  # Convert to percentages

# Create the combined figure
fig = plt.figure(figsize=(16, 12), dpi=300)
gs = GridSpec(3, 3, height_ratios=[1, 1, 2], width_ratios=[1.5, 0.8, 1])

# Panel A: Pascal's Triangle (aligned with Panel B)
ax_triangle = fig.add_subplot(gs[0:2, 0])
for i, row in enumerate(triangle):
    for j, value in enumerate(row):
        ax_triangle.text(j - i / 2, -i, str(value), ha='center', va='center', fontsize=10, weight='bold')
ax_triangle.axis('off')
ax_triangle.set_aspect('equal')
ax_triangle.set_xlim(-5, 10)
ax_triangle.set_ylim(-10, 2)
ax_triangle.set_title("(A) Pascal's Triangle", fontsize=14, weight='bold')

# Panel B: Table
ax_table = fig.add_subplot(gs[0:2, 1])
ax_table.axis('tight')
ax_table.axis('off')

# Create the table
table = ax_table.table(
    cellText=table_data,
    cellLoc='center',
    loc='center',
    colWidths=[0.2, 0.3, 0.3, 0.8],
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2)

# Add a title aligned with others
table_title_ax = fig.add_subplot(gs[0, 1])
table_title_ax.axis('off')  # Hide the axis for title placement
table_title_ax.text(
    0.5, 0.5, "(B) Table of k-Space Contributions", 
    fontsize=14, weight='bold', ha='center', va='center'
)

# Highlight the header row
for (row, col), cell in table.get_celld().items():
    if row == 0:  # Header row
        cell.set_facecolor("#f2f2f2")  # Light grey for header
        cell.set_text_props(weight="bold")

# Panel C: Bar chart of oxidized and reduced percentages
ax_c = fig.add_subplot(gs[2, 0])
ax_c.bar(["Reduced", "Oxidized"], [50, 50], color=["#7EA9E1", "#E07A7A"], alpha=0.9)
ax_c.set_title("(C) Mean Oxidation State Distribution", fontsize=14, weight='bold')
ax_c.set_ylabel("Percentage (%)", fontsize=12, weight='bold')
ax_c.grid(True, linestyle="--", alpha=0.5)

# Panel D: Deconvoluted wave
ax_d = fig.add_subplot(gs[2, 1])
ax_d.plot(x, convoluted_wave, label="Deconvoluted 50%-Oxidized Wave", color="#E97B1E", linewidth=2)
ax_d.set_title("(D) Deconvoluted Waves (k-Space)", fontsize=14, weight='bold')
ax_d.set_xlabel("k-Space", fontsize=12, weight='bold')
ax_d.set_ylabel("Amplitude", fontsize=12, weight='bold')
ax_d.grid(True, linestyle="--", alpha=0.5)
ax_d.legend(fontsize=10, loc="upper right")

# Panel E: Deconvoluted k-Space contributions
ax_e = fig.add_subplot(gs[2, 2])
bars = ax_e.bar(k_values, deconvoluted_contributions, color=["#7EA9E1", "#E6A97E", "#E6A97E", "#B84747"], alpha=0.9)
ax_e.set_title("(E) Deconvoluted k-Space Contributions", fontsize=14, weight='bold')
ax_e.set_xlabel("k-Space", fontsize=12, weight='bold')
ax_e.set_ylabel("Contribution (%)", fontsize=12, weight='bold')
ax_e.set_xticks(k_values)
ax_e.grid(True, linestyle="--", alpha=0.5)
for bar, value in zip(bars, deconvoluted_contributions):
    ax_e.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{value:.1f}%", ha="center", fontsize=10, weight="bold")

# Adjust layout and save
plt.tight_layout()
file_path = "balanced_combined_figure.png"
plt.savefig(file_path, dpi=300)
plt.show()
