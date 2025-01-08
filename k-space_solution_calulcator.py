import itertools
import numpy as np
import pandas as pd

# Proteoforms and their corresponding oxidation states
proteoforms = {
    'alpha': 0,
    'beta': 33.3,
    'gamma': 66.6,
    'delta': 100,  
}

# Target oxidation percentage
target_oxidation = 50

# Fixed number of molecules (e.g., 10 molecules)
num_molecules = 10

# Possible integer counts of molecules in each state (must sum up to num_molecules)
molecule_counts = list(range(num_molecules + 1))

# Function to calculate weighted average
def calculate_weighted_average(proteoform_combination):
    total_molecules = sum(proteoform_combination.values())
    if total_molecules == 0:
        return None  # Prevent division by zero
    oxidation_sum = sum(proteoforms[p] * (count / total_molecules) for p, count in proteoform_combination.items())
    return oxidation_sum

# Generate all possible combinations of counts summing to num_molecules
def generate_valid_combinations():
    for comb in itertools.product(molecule_counts, repeat=len(proteoforms)):
        if sum(comb) == num_molecules:
            yield comb

# Store valid solutions
valid_solutions = []

# Adjusted tolerance for np.isclose()
tolerance = 0.1  # Allow small deviations from the target value

# Check each combination for matching the target oxidation level
for combination in generate_valid_combinations():
    proteoform_combination = dict(zip(proteoforms.keys(), combination))
    weighted_avg = calculate_weighted_average(proteoform_combination)
    if weighted_avg is not None and np.isclose(weighted_avg, target_oxidation, atol=tolerance):
        valid_solutions.append(proteoform_combination)

# Display the valid solutions
df_solutions = pd.DataFrame(valid_solutions)
print("Possible solutions for 10% oxidation state with fixed number of molecules:")
print(df_solutions)

# Save the solutions to a CSV file (optional)
df_solutions.to_csv('oxidation_state_solutions.csv', index=False)
