import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.cm import viridis
from matplotlib.colors import to_hex

# Set font to DejaVu Sans (Colab-friendly alternative to Arial)
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]

# DPI setting for high-resolution figures
dpi = 300

# Function to generate Pascal's Triangle (starting from R = 1)
def generate_pascals_triangle(rows):
    triangle = [[1]]
    for i in range(1, rows):
        row = [1]
        for j in range(1, i):
            row.append(triangle[i-1][j-1] + triangle[i-1][j])
        row.append(1)
        triangle.append(row)
    return triangle

# Generate Pascal's Triangle (1:1 with k-space, starting from R = 1)
rows = 11  # R = 1 to R = 11
triangle = generate_pascals_triangle(rows)

# Define k-space values and their contributions
k_values = np.array([0, 1, 2, 3])
contributions = np.array([0.12, 0.38, 0.38, 0.12])

# Generate x values for the wave
x = np.linspace(-1, 4, 500)

# Function to create the convoluted wave
def create_convoluted_wave(k_values, contributions, x, sigma=0.1):
    return sum(contribution * np.exp(-((x - k) ** 2) / (2 * sigma ** 2)) for k, contribution in zip(k_values, contributions))

# Create the convoluted wave
convoluted_wave = create_convoluted_wave(k_values, contributions, x)

# Generate colors from the Viridis colormap
viridis_colors = [to_hex(viridis(i / 3)) for i in range(4)]

# Create the bar chart data (deconvoluted components)
deconvoluted_contributions = contributions * 100  # Convert to percentages

### **Panel A: Pascal's Triangle**
fig_a, ax_a = plt.subplots(figsize=(6, 7), dpi=dpi)  # Increased figure height for clarity

for i, row in enumerate(triangle[1:]):  # **Start at R = 1**
    y_offset = -i * 1.2  # Adjust row spacing
    for j, value in enumerate(row):
        ax_a.text(j - i / 2, y_offset, str(value), 
                  ha='center', va='center', 
                  fontsize=max(14 - i, 6), weight='bold')  # Adjust font size dynamically

ax_a.axis('off')
ax_a.set_aspect('equal')
ax_a.set_xlim(-6, 6)
ax_a.set_ylim(-rows * 1.2, 2)
ax_a.set_title("(A) Pascal's Triangle (1:1 with k-Space)", fontsize=16, weight='bold')

fig_a.savefig("Panel_A.svg", format="svg", dpi=dpi)

### **Panel B: Table of k-Space Contributions**
fig_b, ax_b = plt.subplots(figsize=(5, 5), dpi=dpi)
ax_b.axis('off')

table_data = [
    ["k", "%-oxidised", "i-states", "Structure"],
    ["0", "0", "1", "[000]"],
    ["1", "33.3", "3", "[100], [010], [001]"],
    ["2", "66.6", "3", "[110], [011], [101]"],
    ["3", "100", "1", "[111]"],
]
table = ax_b.table(
    cellText=table_data,
    cellLoc='center',
    loc='center',
    colWidths=[0.2, 0.3, 0.3, 0.8]
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2)

ax_b.set_title("(B) Table of k-Space Contributions", fontsize=14, weight='bold')
fig_b.savefig("Panel_B.svg", format="svg", dpi=dpi)

### **Panel C: Mean Oxidation State**
fig_c, ax_c = plt.subplots(figsize=(5, 5), dpi=dpi)
ax_c.bar(["Reduced", "Oxidized"], [50, 50], color=[viridis_colors[0], viridis_colors[-1]], alpha=0.9)
ax_c.set_title("(C) Mean Oxidation State", fontsize=14, weight='bold')
ax_c.set_ylabel("Percentage (%)", fontsize=12, weight='bold', labelpad=10)
ax_c.grid(True, linestyle="--", alpha=0.5)
fig_c.savefig("Panel_C.svg", format="svg", dpi=dpi)

### **Panel D: k-Space Wave**
fig_d, ax_d = plt.subplots(figsize=(5, 5), dpi=dpi)
ax_d.plot(x, convoluted_wave, label="Deconvoluted 50%-Oxidized Wave", color=viridis_colors[2], linewidth=2)
ax_d.set_title("(D) k-Space Wave", fontsize=14, weight='bold')
ax_d.set_xlabel("k-Space", fontsize=12, weight='bold', labelpad=10)
ax_d.set_ylabel("Amplitude", fontsize=12, weight='bold', labelpad=10)
ax_d.grid(True, linestyle="--", alpha=0.5)
ax_d.legend(fontsize=10, loc="upper right")
fig_d.savefig("Panel_D.svg", format="svg", dpi=dpi)

### **Panel E: k-Space Contributions**
fig_e, ax_e = plt.subplots(figsize=(5, 5), dpi=dpi)
bars = ax_e.bar(k_values, deconvoluted_contributions, color=viridis_colors, alpha=0.9)
ax_e.set_title("(E) k-Space Contributions", fontsize=14, weight='bold')
ax_e.set_xlabel("k-Space", fontsize=12, weight='bold', labelpad=10)
ax_e.set_ylabel("Contribution (%)", fontsize=12, weight='bold', labelpad=10)
ax_e.set_xticks(k_values)
ax_e.grid(True, linestyle="--", alpha=0.5)
for bar, value in zip(bars, deconvoluted_contributions):
    ax_e.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{value:.1f}%", ha="center", fontsize=10, weight="bold")

fig_e.savefig("Panel_E.svg", format="svg", dpi=dpi)

# Show figures
plt.show()
