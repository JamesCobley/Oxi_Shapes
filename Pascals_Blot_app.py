import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import requests
from itertools import combinations
from scipy.special import comb
from io import BytesIO

def fetch_protein_sequence(uniprot_id):
    """Fetch protein sequence from UniProt"""
    try:
        url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
        response = requests.get(url)
        response.raise_for_status()
        fasta = response.text
        sequence = ''.join(fasta.split('\n')[1:]).replace(' ', '')
        return sequence
    except requests.RequestException as e:
        st.error(f"Failed to fetch data for UniProt ID {uniprot_id}: {e}")
        return None

def calculate_molecular_mass(sequence):
    """Calculate molecular mass of the protein based on the sequence."""
    avg_residue_mass = 0.110  # kDa (110 Da)
    molecular_mass = len(sequence) * avg_residue_mass
    return molecular_mass

def generate_proteoforms(num_cysteines):
    """Generate all unique proteoforms and group them by oxidation state."""
    proteoforms = []
    grouped_proteoforms = [[] for _ in range(num_cysteines + 1)]
    
    for i in range(num_cysteines + 1):
        for combi in combinations(range(num_cysteines), i):
            proteoform = np.zeros(num_cysteines, dtype=int)
            proteoform[list(combi)] = 1
            proteoforms.append(proteoform)
            grouped_proteoforms[i].append(proteoform)
    
    return proteoforms, grouped_proteoforms

def calculate_pascal_row(num_cysteines):
    """Generate the Pascal triangle row for the given number of cysteines."""
    return [int(comb(num_cysteines, k)) for k in range(num_cysteines + 1)]

def predict_band_position(molecular_weight, coefficients):
    """Predict the position of the band using the scaling formula from the standard curve."""
    log_mw = np.log10(molecular_weight)
    pixel_position = np.polyval(coefficients, log_mw)
    return pixel_position

def plot_immunoblot(molecular_mass, grouped_proteoforms, num_cysteines, coefficients):
    """Plot the positions of the redox proteoforms on a scale-invariant simulated immunoblot."""
    
    # Calculate band positions using molecular weights
    band_positions = [molecular_mass + (5 * i) for i in range(len(grouped_proteoforms))]
    
    fig, ax = plt.subplots(figsize=(5, 8))

    # Plot the redox proteoforms at their corresponding molecular weights
    for i, pos in enumerate(band_positions):
        y_pos = predict_band_position(pos, coefficients)  # Position the band using the scaling
        band_intensity = (num_cysteines - i + 1) / (num_cysteines + 1)  # Intensity decreases with oxidation
        ax.plot([0.3, 0.7], [y_pos, y_pos], linewidth=10 * band_intensity, color='black')
        ax.text(0.75, y_pos, f'{(100 * (num_cysteines - i) / num_cysteines):.1f}%', verticalalignment='center', fontsize=12)

    # Set fixed y-axis limits (scale invariant)
    ax.set_ylim(predict_band_position(250, coefficients), predict_band_position(10, coefficients))
    
    ax.set_xlim(0, 1)
    ax.set_yticks([predict_band_position(mw, coefficients) for mw in [10, 25, 37, 50, 75, 100, 150, 250]])
    ax.set_yticklabels([f'{mw:.0f} kDa' for mw in [10, 25, 37, 50, 75, 100, 150, 250]])
    ax.set_xticks([])
    ax.set_xlabel('Protein Redox States', fontsize=15)
    ax.set_ylabel('Molecular Mass (kDa)', fontsize=15)
    ax.set_title('Simulated Immunoblot', fontsize=20)

    # Invert the y-axis to match the appearance of a real blot
    ax.invert_yaxis()

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    return buf

# Streamlit app
st.title("Pascal\'s Blot Simulation App")

uniprot_id = st.text_input("Enter UniProt Accession Number:", "P04406")

if uniprot_id:
    sequence = fetch_protein_sequence(uniprot_id)
    if sequence:
        num_cysteines = sequence.count('C')  # Count the number of cysteines
        molecular_mass = calculate_molecular_mass(sequence)
        
        # Calculate the molecular mass of the 100%-oxidised form
        oxidised_mass = molecular_mass + (num_cysteines * 5)

        # 1. Number of bands is the cysteine residue count + 1
        num_bands = num_cysteines + 1
        st.write(f"Number of Bands: {num_bands}")
        
        # 2. Number of cysteine redox proteoforms (2^num_cysteines)
        num_proteoforms = 2 ** num_cysteines
        st.write(f"i-space: {num_proteoforms}")

        # 3. Pascal triangle-based proteoform structure
        pascal_row = calculate_pascal_row(num_cysteines)
        st.write(f"k-space: {pascal_row}")

        st.write(f"Protein Sequence Length: {len(sequence)} amino acids")
        st.write(f"Molecular Mass (Reduced Form): {molecular_mass:.2f} kDa")
        st.write(f"Molecular Mass (100%-Oxidised Form): {oxidised_mass:.2f} kDa")
        st.write(f"R: {num_cysteines}")

        # Cleland Immunoblot suitability
        if oxidised_mass < 152:
            st.success("Yes, this protein is a good candidate for Pascal\'s Blot.")
        else:
            st.warning("No, this protein is not a good candidate for Pascal\'s Blot.")

        if num_cysteines > 0:
            # Coefficients from the standard curve
            coefficients = [-0.00501309, 2.38407094]  # Replace with real coefficients
            _, grouped_proteoforms = generate_proteoforms(num_cysteines)
            buf = plot_immunoblot(molecular_mass, grouped_proteoforms, num_cysteines, coefficients)

            if buf:
                # Display the plot
                st.image(buf, use_column_width=True, caption='Simulated Pascal's Blot')

                # Download button
                buf.seek(0)
                st.download_button(
                    label="Download Immunoblot",
                    data=buf,
                    file_name="Simulated_Immunoblot.png",
                    mime="image/png"
                )
