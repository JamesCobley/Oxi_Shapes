import itertools
import pandas as pd
import math
from scipy.special import comb  # For binomial coefficients
from google.colab import files  # For downloading the Excel file

# Function to generate all binary proteoforms based on the number of cysteines (R)
def generate_proteoforms(R):
    """Generate binary proteoforms for a given number of cysteines (R)."""
    num_states = 2**R  # Total number of proteoforms
    proteoforms = [format(i, f'0{R}b') for i in range(num_states)]  # Binary format strings
    return proteoforms

# Function to find allowed transitions (±1 oxidation state changes)
def find_allowed_transitions(proteoform, proteoforms, R):
    """Find allowed transitions by flipping one bit at a time."""
    allowed = [proteoform]  # Self-transition is always allowed
    struct_vec = list(proteoform)  # Convert string to list for modification

    # Current k-value (number of oxidized cysteines)
    current_k = struct_vec.count('1')

    # Generate allowed transitions by flipping one bit at a time
    for i in range(R):
        new_structure = struct_vec[:]
        new_structure[i] = '1' if new_structure[i] == '0' else '0'  # Toggle oxidation state
        new_proteoform = ''.join(new_structure)
        new_k = new_proteoform.count('1')

        # Only allow transitions where k changes by ±1
        if abs(new_k - current_k) == 1:
            allowed.append(new_proteoform)

    return allowed

# Function to find barred transitions (all states not in the allowed set)
def find_barred_transitions(proteoform, allowed_transitions, proteoforms):
    """Find barred transitions by excluding allowed ones."""
    return [pf for pf in proteoforms if pf not in allowed_transitions]

# Function to compute Pascal's row for a given R
def pascal_row(R):
    """Compute Pascal's Triangle row for a given R (binomial coefficients)."""
    return [int(comb(R, k)) for k in range(R + 1)]

# Function to generate full transition dataset and save as Excel file
def generate_transition_data(R, file_name="proteoform_transitions.xlsx"):
    """Generate proteoform transitions and save to an Excel file."""
    proteoforms = generate_proteoforms(R)
    data = []
    
    # Compute Pascal's row (binomial coefficients)
    binomial_coefficients = pascal_row(R)

    for proteoform in proteoforms:
        k_value = proteoform.count('1')  # Number of oxidized cysteines
        percent_ox = (k_value / R) * 100 if R > 0 else 0  # Percentage oxidation
        allowed_transitions = find_allowed_transitions(proteoform, proteoforms, R)
        barred_transitions = find_barred_transitions(proteoform, allowed_transitions, proteoforms)

        # Conservation of Degrees calculations
        k_values = [pf.count('1') for pf in allowed_transitions]
        K_minus_0 = sum(1 for k in k_values if k < k_value)
        K_plus = sum(1 for k in k_values if k > k_value)
        conservation_of_degrees = K_minus_0 + K_plus + 1  # Including self-transition

        # Store results
        data.append({
            "Proteoform": proteoform,
            "k-Value": k_value,
            "Percent Oxidation": percent_ox,
            "Allowed Transitions": ", ".join(allowed_transitions),
            "Barred Transitions": ", ".join(barred_transitions),
            "K_minus_0": K_minus_0,
            "K_plus": K_plus,
            "Conservation of Degrees": conservation_of_degrees
        })

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Create a summary DataFrame for i-space and k-space cardinalities
    summary_df = pd.DataFrame({
        "i-Space Cardinality": [2**R],
        "k-Space Cardinality": [R + 1],
        "Pascal Row": [", ".join(map(str, binomial_coefficients))]  # **Fixed Formatting**
    })

    # Save to Excel
    with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name="Proteoform Transitions", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"Excel file saved successfully as {file_name}")

    return df, summary_df, file_name

# Run the script with user input for R
R = int(input("Enter the number of cysteines (R): "))
file_name = input("Enter the file name (with .xlsx extension) to save the output: ")

# Generate and save the transition data
df, summary_df, file_path = generate_transition_data(R, file_name)

# Provide a download link
files.download(file_path)

# Display summary data
print("\nSummary Data:")
print(summary_df)
