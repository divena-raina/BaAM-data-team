"""
Add shooting-related homicide flags to the aggregated dataset.

This script:
1. Loads individual homicide records
2. Matches them with shooting records by CASE_NUMBER
3. Flags which homicides involved firearms
4. Aggregates to community area × year level
5. Adds shooting-related homicide counts to the master aggregated dataset
"""

import pandas as pd
import numpy as np

# Load the datasets
print("Loading datasets...")
homicides = pd.read_csv('/Users/elyssaqi/Desktop/ura-2/BaAM-data-team/cleaned_data/Chicago_Homicides_Cleaned.csv')
shootings = pd.read_csv('/Users/elyssaqi/Desktop/ura-2/BaAM-data-team/cleaned_data/shooting dataset pt3/cleaned_shooting_dataset_pt3.csv')
aggregated = pd.read_csv('/Users/elyssaqi/Desktop/ura-2/BaAM-data-team/merged data sets/aggregated/aggregated_chicago_dataset_2012plus.csv')

print(f"Loaded {len(homicides)} homicides")
print(f"Loaded {len(shootings)} shooting incidents")
print(f"Loaded {len(aggregated)} aggregated records")

# Standardize case numbers for matching
print("\nStandardizing case numbers...")
homicides['case_number_clean'] = homicides['case_number'].str.strip().str.upper()
shootings['CASE_NUMBER_clean'] = shootings['CASE_NUMBER'].str.strip().str.upper()

# Get unique shooting case numbers
shooting_cases = set(shootings['CASE_NUMBER_clean'].unique())
print(f"Found {len(shooting_cases)} unique shooting case numbers")

# Flag homicides that appear in shootings dataset (firearm-related)
homicides['is_shooting'] = homicides['case_number_clean'].isin(shooting_cases).astype(int)

# Filter to 2012+ to match aggregated dataset
homicides_2012plus = homicides[homicides['year'] >= 2012].copy()

print(f"\nHomicides 2012+: {len(homicides_2012plus)}")
print(f"Firearm-related homicides: {homicides_2012plus['is_shooting'].sum()}")
print(f"Non-firearm homicides: {(homicides_2012plus['is_shooting'] == 0).sum()}")
print(f"Percentage firearm-related: {homicides_2012plus['is_shooting'].mean() * 100:.1f}%")

# Aggregate by community area and year - check if ANY homicide was firearm-related
print("\nAggregating by community area and year...")
shooting_flag = homicides_2012plus.groupby(['community_area', 'year']).agg({
    'is_shooting': 'max'  # 1 if ANY homicide in that area/year was firearm-related, else 0
}).reset_index()

# Merge with aggregated dataset
print("Merging with aggregated dataset...")
aggregated_with_shooting = aggregated.merge(
    shooting_flag,
    on=['community_area', 'year'],
    how='left'
)

# Fill NaN values with 0 (areas with no homicides)
aggregated_with_shooting['is_shooting'] = aggregated_with_shooting['is_shooting'].fillna(0).astype(int)

print("\nData validation:")
print(f"Total rows in output: {len(aggregated_with_shooting)}")
print(f"Rows with is_shooting = 1: {(aggregated_with_shooting['is_shooting'] == 1).sum()}")
print(f"Rows with is_shooting = 0: {(aggregated_with_shooting['is_shooting'] == 0).sum()}")

# Display summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"\nRows with firearm-related homicides (is_shooting=1): {(aggregated_with_shooting['is_shooting'] == 1).sum()}")
print(f"Rows with NO firearm-related homicides (is_shooting=0): {(aggregated_with_shooting['is_shooting'] == 0).sum()}")
print(f"Percentage of area/year combinations with firearm homicides: {aggregated_with_shooting['is_shooting'].mean() * 100:.1f}%")

# Show sample of results
print("\n" + "="*60)
print("SAMPLE OF RESULTS")
print("="*60)
print("\nSample rows with is_shooting flag:")
sample = aggregated_with_shooting[['community_area', 'year', 'homicide_count', 'is_shooting']].head(20)
print(sample.to_string(index=False))

# Save the updated dataset
output_path = '/Users/elyssaqi/Desktop/ura-2/BaAM-data-team/merged data sets/aggregated/aggregated_with_shooting_flag.csv'
aggregated_with_shooting.to_csv(output_path, index=False)
print(f"\n✅ Saved updated dataset to: {output_path}")

# Also save a data dictionary
print("\nCreating data dictionary...")
dict_data = []
for col in aggregated_with_shooting.columns:
    dict_data.append({
        'Column_Name': col,
        'Data_Type': str(aggregated_with_shooting[col].dtype),
        'Non_Null_Count': aggregated_with_shooting[col].notna().sum(),
        'Unique_Values': aggregated_with_shooting[col].nunique(),
        'Sample_Value': aggregated_with_shooting[col].dropna().iloc[0] if len(aggregated_with_shooting[col].dropna()) > 0 else None,
        'Description': '1 if any homicide in this area/year was firearm-related, 0 otherwise' if col == 'is_shooting' else ''
    })

dict_df = pd.DataFrame(dict_data)
dict_path = '/Users/elyssaqi/Desktop/ura-2/BaAM-data-team/merged data sets/aggregated/aggregated_with_shooting_flag_dictionary.csv'
dict_df.to_csv(dict_path, index=False)
print(f"✅ Saved data dictionary to: {dict_path}")

print("\n" + "="*60)
print("COMPLETE!")
print("="*60)

