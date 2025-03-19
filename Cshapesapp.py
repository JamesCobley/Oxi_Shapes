import streamlit as st
import pandas as pd
import math
from scipy.special import comb  # For binomial coefficients
import io

# Function to generate binary proteoforms based on the number of cysteines (R)
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
def generate_transition_data(R):
    """Generate proteoform transitions and return data as DataFrame."""
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
        "Pascal Row": [", ".join(map(str, binomial_coefficients))]  # Proper Formatting
    })

    return df, summary_df

# Function to create an Excel file and return it as a downloadable object
def create_excel(df, summary_df):
    """Generate an Excel file as a downloadable object."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name="Proteoform Transitions", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
    
    output.seek(0)  # Reset pointer
    return output

# -------------- STREAMLIT APP --------------
st.title("Proteoform Transitions Explorer")

# Input: Number of Cysteines (R)
R = st.number_input("Enter the number of cysteines (R)", min_value=1, max_value=50, value=10, step=1)

# Button to compute
if st.button("Compute Transitions"):
    df, summary_df = generate_transition_data(R)

    # Display Summary Data
    st.subheader("Summary Data")
    st.write(summary_df)

    # Display Main Transition Table
    st.subheader("Proteoform Transitions")
    st.write(df)

    # Create downloadable Excel file
    excel_data = create_excel(df, summary_df)
    st.download_button(
        label="Download Excel File",
        data=excel_data,
        file_name=f"proteoform_transitions_R{R}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
