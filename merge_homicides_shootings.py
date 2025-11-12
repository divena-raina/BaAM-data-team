"""
Merge Master Homicides Dataset with Shootings Dataset
Adds is_shooting binary flag to indicate if a homicide involved a firearm
"""

import pandas as pd
import os

# Define file paths
homicides_path = 'cleaned_data/Chicago_Homicides_Cleaned.csv'
shootings_path = 'homicide_datasets/prev sprint/Chicago_Shooting_Dataset.csv'

# Read the datasets
print("Reading homicides dataset...")
homicides_df = pd.read_csv(homicides_path, low_memory=False)
print(f"Homicides dataset shape: {homicides_df.shape}")

print("\nReading shootings dataset...")
shootings_df = pd.read_csv(shootings_path, low_memory=False)
print(f"Shootings dataset shape: {shootings_df.shape}")

# Standardize case number column names for comparison
# Homicides uses 'case_number', shootings uses 'Case Number'
homicides_case_col = 'case_number'
shootings_case_col = 'Case Number'

# Get unique case numbers from shootings dataset (strip whitespace for safety)
shooting_case_numbers = set(shootings_df[shootings_case_col].dropna().astype(str).str.strip())
print(f"\nUnique case numbers in shootings dataset: {len(shooting_case_numbers)}")

# Create is_shooting flag: 1 if case_number appears in shootings, else 0
# Strip whitespace from homicide case numbers for comparison
print("\nCreating is_shooting flag...")
homicides_df['is_shooting'] = homicides_df[homicides_case_col].astype(str).str.strip().isin(shooting_case_numbers).astype(int)

# Display summary statistics
print("\n" + "="*60)
print("MERGE SUMMARY")
print("="*60)
total_homicides = len(homicides_df)
shooting_homicides = homicides_df['is_shooting'].sum()
non_shooting_homicides = total_homicides - shooting_homicides

print(f"Total homicides: {total_homicides:,}")
print(f"Homicides involving firearms (is_shooting=1): {shooting_homicides:,} ({shooting_homicides/total_homicides*100:.2f}%)")
print(f"Homicides not involving firearms (is_shooting=0): {non_shooting_homicides:,} ({non_shooting_homicides/total_homicides*100:.2f}%)")

# Show sample of merged data
print("\n" + "="*60)
print("SAMPLE OF MERGED DATA (first 10 rows with is_shooting=1)")
print("="*60)
sample_shooting = homicides_df[homicides_df['is_shooting'] == 1][['case_number', 'date', 'description', 'is_shooting']].head(10)
print(sample_shooting.to_string(index=False))

print("\n" + "="*60)
print("SAMPLE OF MERGED DATA (first 10 rows with is_shooting=0)")
print("="*60)
sample_non_shooting = homicides_df[homicides_df['is_shooting'] == 0][['case_number', 'date', 'description', 'is_shooting']].head(10)
print(sample_non_shooting.to_string(index=False))

# Save the merged dataset
output_path = 'cleaned_data/Chicago_Homicides_Cleaned.csv'
print(f"\nSaving merged dataset to: {output_path}")
homicides_df.to_csv(output_path, index=False)
print("Merge complete! Dataset saved successfully.")

