import itertools
import pandas as pd

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

# Function to generate full transition dataset and save as Excel file
def generate_transition_data(R, file_name="proteoform_transitions.xlsx"):
    """Generate proteoform transitions and save to an Excel file."""
    proteoforms = generate_proteoforms(R)
    data = []

    for proteoform in proteoforms:
        k_value = proteoform.count('1')  # Number of oxidized cysteines
        percent_ox = (k_value / R) * 100  # Percentage oxidation
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

    # Save to Excel
    df.to_excel(file_name, index=False)
    print(f"Excel file saved successfully as {file_name}")

    return df, file_name

# Run the script with user input for R
R = int(input("Enter the number of cysteines (R): "))
file_name = input("Enter the file name (with .xlsx extension) to save the output: ")

# Generate and save the transition data
df, file_path = generate_transition_data(R, file_name)

# Provide a download link
from google.colab import files
files.download(file_path)

# Display the first few rows of the DataFrame
df.head()
