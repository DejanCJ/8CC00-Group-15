import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors

# Function to calculate descriptors for a single molecule
def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * len(descriptor_names)
    return calculator.CalcDescriptors(mol)

# Read the input CSV file
input_csv = "C:/Users/20192891/Documents/Master/Q4/8CC00/Assignment 3/tested_molecules.csv"  # Replace with your input file path
df = pd.read_csv(input_csv)

# Ensure the CSV has a column named 'SMILES'
if 'SMILES' not in df.columns:
    raise ValueError("Input CSV file must contain a 'SMILES' column.")

# Get list of descriptor names
descriptor_names = [desc[0] for desc in Descriptors._descList]

# Create a MolecularDescriptorCalculator
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

# Calculate descriptors for each SMILES string and store in a new DataFrame
descriptors_df = pd.DataFrame(df['SMILES'].apply(calculate_descriptors).tolist(), columns=descriptor_names)

# Check for empty cells (NaNs)
if descriptors_df.isnull().values.any():
    print("Warning: There are empty cells in the descriptors data.")

# Concatenate the original DataFrame with the descriptors DataFrame
result_df = pd.concat([df, descriptors_df], axis=1)

# Log removed descriptors
removed_descriptors = []

# Filter out descriptors with low variance
def low_variance_filter(df, threshold=0.01):
    variances = df.var()
    low_variance_cols = variances[variances <= threshold].index
    removed_descriptors.extend([(col, 'Low Variance') for col in low_variance_cols])
    return df.loc[:, variances > threshold]

# Filter out highly correlated descriptors, keeping the one with higher variance
def high_correlation_filter(df, threshold=0.90):
    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = set()
    for column in upper_triangle.columns:
        for row in upper_triangle.index:
            if upper_triangle.at[row, column] > threshold:
                # Compare variances and keep the one with higher variance
                if df[column].var() >= df[row].var():
                    to_drop.add(row)
                else:
                    to_drop.add(column)

    removed_descriptors.extend([(col, 'High Correlation') for col in to_drop])
    return df.drop(columns=to_drop)

# Apply filters
descriptors_df_filtered = low_variance_filter(descriptors_df)
descriptors_df_filtered = high_correlation_filter(descriptors_df_filtered)

# Concatenate the original DataFrame with the filtered descriptors DataFrame
result_df_filtered = pd.concat([df, descriptors_df_filtered], axis=1)

# Write the result to a new CSV file
output_csv = 'C:/Users/20192891/Documents/Master/Q4/8CC00/Assignment 3/output_descriptors_filtered.csv'  # Replace with your desired output file path
result_df_filtered.to_csv(output_csv, index=False)

# Write removed descriptors to a log file
log_file = 'C:/Users/20192891/Documents/Master/Q4/8CC00/Assignment 3/removed_descriptors_log.csv'
removed_df = pd.DataFrame(removed_descriptors, columns=['Descriptor', 'Reason'])
removed_df.to_csv(log_file, index=False)

print(f"Descriptors calculated, filtered, and saved to {output_csv}")
print(f"Removed descriptors log saved to {log_file}")
