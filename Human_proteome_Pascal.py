import gzip
from Bio import SeqIO
import pandas as pd
from Bio.SeqUtils import molecular_weight

# Function to calculate cysteine count and molecular mass
def get_protein_info(seq_record):
    sequence = str(seq_record.seq)
    cysteine_count = sequence.count('C')  # Count cysteines ('C')
    mol_mass = molecular_weight(sequence, seq_type='protein') / 1000  # Molecular mass in kDa
    return cysteine_count, mol_mass

# Path to gzipped FASTA file
fasta_gz_path = '/content/proteins.fasta.gz'  # Replace with your FASTA file path

# List to store protein data
protein_data = []

# Open the gzipped FASTA file and read protein sequences
with gzip.open(fasta_gz_path, "rt") as fasta_file:
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        protein_id = seq_record.id  # UniProt ID (from the FASTA header)
        cysteine_count, mol_mass = get_protein_info(seq_record)  # Get cysteine count and molecular mass
        
        # Calculate the oxidized molecular mass
        oxidized_mass = mol_mass + (cysteine_count * 5)
        
        # Add the protein information to the list
        protein_data.append({
            'Protein_ID': protein_id,
            'Cysteine Residue Integer': cysteine_count,
            '100%-Reduced_Molecular_Mass': mol_mass,
            '100%-Oxidised_Molecular_Mass': oxidized_mass
        })

# Create a DataFrame from the collected data
protein_df = pd.DataFrame(protein_data)

# Write the DataFrame to an Excel file
output_file = '/content/protein_masses.xlsx'  # Path to save the Excel file
protein_df.to_excel(output_file, index=False)

print(f"Excel file with protein data saved to: {output_file}")
