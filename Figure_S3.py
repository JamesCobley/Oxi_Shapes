import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file into a DataFrame
input_file = '/content/protein_masses.xlsx'  # Path to the Excel file
protein_df = pd.read_excel(input_file)

# (a) Exclude all cysteine residue integer values = 0
filtered_df = protein_df[protein_df['Cysteine Residue Integer'] > 0].copy()  # Exclude proteins with 0 cysteines

# (b) Create a histogram for the distribution of proteins by 100%-Reduced Molecular Weight
plt.figure(figsize=(10, 6))
plt.hist(filtered_df['100%-Reduced_Molecular_Mass'], bins=30, color='skyblue', edgecolor='black')
plt.xlabel('100%-Reduced Molecular Weight (kDa)', fontsize=14)
plt.ylabel('Number of Proteins', fontsize=14)
plt.title('Distribution of Proteins by 100%-Reduced Molecular Weight', fontsize=16)
plt.xlim(0, 500)  # Limit the x-axis to 500 kDa to zoom in on the relevant range
plt.tight_layout()
plt.savefig('/content/histogram_reduced_molecular_weight_scaled.png', dpi=300)
plt.show()

# (c) Create a histogram for the distribution of proteins by 100%-Oxidised Molecular Weight
plt.figure(figsize=(10, 6))
plt.hist(filtered_df['100%-Oxidised_Molecular_Mass'], bins=30, color='skyblue', edgecolor='black')
plt.xlabel('100%-Oxidised Molecular Weight (kDa)', fontsize=14)
plt.ylabel('Number of Proteins', fontsize=14)
plt.title('Distribution of Proteins by 100%-Oxidised Molecular Weight', fontsize=16)
plt.xlim(0, 500)  # Limit the x-axis to 500 kDa to zoom in on the relevant range
plt.tight_layout()
plt.savefig('/content/histogram_oxidised_molecular_weight_scaled.png', dpi=300)
plt.show()

# (d) Generate a bar graph showing the percentage of detectable proteins with cutoffs at 150 kDa and 200 kDa
# Detectable proteins for 150 kDa cutoff
detectable_150_df = filtered_df[filtered_df['100%-Oxidised_Molecular_Mass'] <= 150]
detectable_150_percentage = (len(detectable_150_df) / len(filtered_df)) * 100

# Detectable proteins for 200 kDa cutoff
detectable_200_df = filtered_df[filtered_df['100%-Oxidised_Molecular_Mass'] <= 200]
detectable_200_percentage = (len(detectable_200_df) / len(filtered_df)) * 100

# Bar plot for percentages
plt.figure(figsize=(8, 6))
plt.bar(['150 kDa', '200 kDa'], [detectable_150_percentage, detectable_200_percentage], color='skyblue')
plt.xlabel('Mass Cutoff', fontsize=14)
plt.ylabel('Percentage of Detectable Proteins', fontsize=14)
plt.title('Percentage of Detectable Proteins by Mass Cutoff', fontsize=16)
plt.tight_layout()
plt.savefig('/content/detectable_proteins_cutoff.png', dpi=300)
plt.show()

# (e) Show the number of proteins in each cysteine residue range (1-20) and the number of detectable proteins
# Filter for proteins with cysteine residue count between 1 and 20
range_df = filtered_df[filtered_df['Cysteine Residue Integer'].between(1, 20)]

# Total number of proteins per cysteine residue count
total_per_cysteine_count = range_df['Cysteine Residue Integer'].value_counts().sort_index()

# Number of detectable proteins per cysteine residue count for both cutoffs
detectable_per_cysteine_150 = range_df[range_df['100%-Oxidised_Molecular_Mass'] <= 150]['Cysteine Residue Integer'].value_counts().sort_index()
detectable_per_cysteine_200 = range_df[range_df['100%-Oxidised_Molecular_Mass'] <= 200]['Cysteine Residue Integer'].value_counts().sort_index()

# Prepare a DataFrame for plotting
detectable_df = pd.DataFrame({
    'Total Proteins': total_per_cysteine_count,
    'Detectable (≤150 kDa)': detectable_per_cysteine_150,
    'Detectable (≤200 kDa)': detectable_per_cysteine_200
}).fillna(0)  # Fill missing values with 0 for bins with no detectable proteins

# Plot the data
detectable_df.plot(kind='bar', figsize=(12, 6), color=['skyblue', 'orange', 'green'], edgecolor='black')
plt.xlabel('Cysteine Residue Count (1-20)', fontsize=14)
plt.ylabel('Number of Proteins', fontsize=14)
plt.title('Proteins and Detectable Proteins by Cysteine Residue Count', fontsize=16)
plt.tight_layout()
plt.savefig('/content/proteins_per_cysteine_count.png', dpi=300)
plt.show()

print("All plots, including scaled histograms, have been saved successfully.")
