"""
Create Aggregated (2012+) and Comprehensive Unaggregated Datasets
================================================================
This script creates two versions:
1. Aggregated data (2012+ only) - current format but filtered
2. Comprehensive unaggregated data - all individual records from all 5 datasets

Author: Data Team
Date: October 1, 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Community Area Name to Number Mapping (Chicago's 77 official community areas)
COMMUNITY_AREA_MAPPING = {
    'Rogers Park': 1, 'West Ridge': 2, 'Uptown': 3, 'Lincoln Square': 4,
    'North Center': 5, 'Lake View': 6, 'Lincoln Park': 7, 'Near North Side': 8,
    'Edison Park': 9, 'Norwood Park': 10, 'Jefferson Park': 11, 'Forest Glen': 12,
    'North Park': 13, 'Albany Park': 14, 'Portage Park': 15, 'Irving Park': 16,
    'Dunning': 17, 'Montclaire': 18, 'Belmont Cragin': 19, 'Hermosa': 20,
    'Avondale': 21, 'Logan Square': 22, 'Humboldt park': 23, 'West Town': 24,
    'Austin': 25, 'West Garfield Park': 26, 'East Garfield Park': 27, 'Near West Side': 28,
    'North Lawndale': 29, 'South Lawndale': 30, 'Lower West Side': 31, 'Loop': 32,
    'Near South Side': 33, 'Armour Square': 34, 'Douglas': 35, 'Oakland': 36,
    'Fuller Park': 37, 'Grand Boulevard': 38, 'Kenwood': 39, 'Washington Park': 40,
    'Hyde Park': 41, 'Woodlawn': 42, 'South Shore': 43, 'Chatham': 44,
    'Avalon Park': 45, 'South Chicago': 46, 'Burnside': 47, 'Calumet Heights': 48,
    'Roseland': 49, 'Pullman': 50, 'South Deering': 51, 'East Side': 52,
    'West Pullman': 53, 'Riverdale': 54, 'Hegewisch': 55, 'Garfield Ridge': 56,
    'Archer Heights': 57, 'Brighton Park': 58, 'McKinley Park': 59, 'Bridgeport': 60,
    'New City': 61, 'West Elsdon': 62, 'Gage Park': 63, 'Clearing': 64,
    'West Lawn': 65, 'Chicago Lawn': 66, 'West Englewood': 67, 'Englewood': 68,
    'Greater Grand Crossing': 69, 'Ashburn': 70, 'Auburn Gresham': 71, 'Beverly': 72,
    'Washington Height': 73, 'Mount Greenwood': 74, 'Morgan Park': 75, 'O\'Hare': 76,
    "O'Hare": 76, 'Edgewater': 77
}

print("=" * 80)
print("CREATING AGGREGATED (2012+) AND COMPREHENSIVE UNAGGREGATED DATASETS")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD ALL DATASETS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 1: Loading All Source Datasets")
print("=" * 80)

# Load all datasets
print("\n[1/5] Loading Homicides...")
df_homicides = pd.read_csv('cleaned_data/Chicago_Homicides_Cleaned.csv')
print(f"   ✓ Loaded {len(df_homicides):,} homicide records")

print("\n[2/5] Loading Shooting Crimes...")
df_shootings = pd.read_csv('homicide_datasets/Chicago_Shooting_Crimes_Filtered_2012Plus.csv')
print(f"   ✓ Loaded {len(df_shootings):,} shooting records")

print("\n[3/5] Loading Sociodemographic Features...")
df_socio = pd.read_csv('socio_data/sprint 6&7/sociodemographic_features_by_area.csv')
print(f"   ✓ Loaded {len(df_socio):,} community area records")

print("\n[4/5] Loading Population Counts...")
df_population = pd.read_csv('population_datasets/Chicago_Population_Counts_Cleaned.csv')
print(f"   ✓ Loaded {len(df_population):,} population records")

print("\n[5/5] Loading Business Licenses...")
df_business = pd.read_csv('business_licenses_datasets/business_licenses.csv')
print(f"   ✓ Loaded {len(df_business):,} business license records")

# ============================================================================
# STEP 2: CREATE AGGREGATED DATASET (2012+ ONLY)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: Creating Aggregated Dataset (2012+ Only)")
print("=" * 80)

# Standardize column names
def standardize_columns(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
    return df

df_homicides = standardize_columns(df_homicides)
df_shootings = standardize_columns(df_shootings)
df_socio = standardize_columns(df_socio)
df_population = standardize_columns(df_population)
df_business = standardize_columns(df_business)

# Process community areas and years
if 'community_area' in df_homicides.columns:
    df_homicides['community_area'] = pd.to_numeric(df_homicides['community_area'], errors='coerce').astype('Int64')
if 'community_area' in df_shootings.columns:
    df_shootings['community_area'] = pd.to_numeric(df_shootings['community_area'], errors='coerce').astype('Int64')
if 'community_area' in df_socio.columns:
    df_socio['community_area_num'] = df_socio['community_area'].map(COMMUNITY_AREA_MAPPING)
    df_socio.rename(columns={'community_area': 'community_area_name', 
                             'community_area_num': 'community_area'}, inplace=True)
if 'community_area_number' in df_population.columns:
    df_population.rename(columns={'community_area_number': 'community_area'}, inplace=True)
    df_population['community_area'] = pd.to_numeric(df_population['community_area'], errors='coerce').astype('Int64')
if 'community_area' in df_business.columns:
    df_business['community_area'] = pd.to_numeric(df_business['community_area'], errors='coerce').astype('Int64')

# Convert years
for df in [df_homicides, df_shootings, df_population]:
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')

# Filter to 2012+ only
print("\n🔍 Filtering to 2012+ data...")
df_homicides_2012 = df_homicides[df_homicides['year'] >= 2012].copy()
df_shootings_2012 = df_shootings[df_shootings['year'] >= 2012].copy()
df_population_2012 = df_population[df_population['year'] >= 2012].copy()

print(f"   Homicides 2012+: {len(df_homicides_2012):,} records")
print(f"   Shootings 2012+: {len(df_shootings_2012):,} records")
print(f"   Population 2012+: {len(df_population_2012):,} records")

# Aggregate homicides (2012+)
print("\n🔄 Aggregating homicides 2012+...")
homicides_agg = df_homicides_2012.groupby(['community_area', 'year'], dropna=False).agg({
    'id': 'count',
    'arrest': lambda x: x.sum() if x.notna().any() else 0,
    'domestic': lambda x: x.sum() if x.notna().any() else 0
}).rename(columns={
    'id': 'homicide_count',
    'arrest': 'homicide_arrests',
    'domestic': 'homicide_domestic'
}).reset_index()

# Aggregate shootings (2012+)
print("\n🔄 Aggregating shootings 2012+...")
shootings_agg = df_shootings_2012.groupby(['community_area', 'year'], dropna=False).agg({
    'id': 'count',
    'arrest': lambda x: x.sum() if x.notna().any() else 0,
    'domestic': lambda x: x.sum() if x.notna().any() else 0
}).rename(columns={
    'id': 'shooting_count',
    'arrest': 'shooting_arrests',
    'domestic': 'shooting_domestic'
}).reset_index()

# Aggregate business licenses (2012+)
print("\n🔄 Aggregating business licenses 2012+...")
if 'license_term_start_date' in df_business.columns:
    df_business['license_date'] = pd.to_datetime(df_business['license_term_start_date'], errors='coerce')
    df_business['license_year'] = df_business['license_date'].dt.year
    df_business_2012 = df_business[df_business['license_year'] >= 2012].copy()
    
    business_agg = df_business_2012.dropna(subset=['community_area', 'license_year']).groupby(
        ['community_area', 'license_year']
    ).agg({
        'id': 'count',
        'license_description': 'nunique'
    }).rename(columns={
        'id': 'business_license_count',
        'license_description': 'unique_business_types'
    }).reset_index()
    business_agg.rename(columns={'license_year': 'year'}, inplace=True)
    business_agg['year'] = business_agg['year'].astype('Int64')
else:
    business_agg = pd.DataFrame(columns=['community_area', 'year'])

# Prepare population data (2012+)
pop_cols = ['community_area', 'year']
pop_data_cols = [col for col in df_population_2012.columns if 'population' in col.lower()]
population_subset = df_population_2012[pop_cols + pop_data_cols[:7]].copy()
population_subset.columns = [col.replace('population___', 'pop_').replace('population__', 'pop_').replace('_', '_').lower() for col in population_subset.columns]

# Create base grid for 2012+
years_2012 = sorted([y for y in range(2012, 2026)])
community_areas = sorted([i for i in range(1, 78)])

print(f"\n📊 Creating 2012+ grid: {len(community_areas)} areas × {len(years_2012)} years = {len(community_areas) * len(years_2012):,} records")
base_grid_2012 = pd.DataFrame([(ca, yr) for ca in community_areas for yr in years_2012], 
                              columns=['community_area', 'year'])
base_grid_2012['community_area'] = base_grid_2012['community_area'].astype('Int64')
base_grid_2012['year'] = base_grid_2012['year'].astype('Int64')

# Merge aggregated data
print("\n🔗 Merging aggregated 2012+ data...")
aggregated_df = base_grid_2012.merge(homicides_agg, on=['community_area', 'year'], how='left')
aggregated_df = aggregated_df.merge(shootings_agg, on=['community_area', 'year'], how='left')
aggregated_df = aggregated_df.merge(population_subset, on=['community_area', 'year'], how='left')
if len(business_agg) > 0:
    aggregated_df = aggregated_df.merge(business_agg, on=['community_area', 'year'], how='left')

# Add sociodemographics
socio_clean = df_socio[[col for col in df_socio.columns if col != 'community_area_name']].copy()
aggregated_df = aggregated_df.merge(socio_clean, on='community_area', how='left')

# Fill missing values
crime_cols = ['homicide_count', 'homicide_arrests', 'homicide_domestic',
              'shooting_count', 'shooting_arrests', 'shooting_domestic',
              'business_license_count', 'unique_business_types']

for col in crime_cols:
    if col in aggregated_df.columns:
        aggregated_df[col] = aggregated_df[col].fillna(0)

# Interpolate population
pop_cols_in_df = [col for col in aggregated_df.columns if col.startswith('pop_')]
if pop_cols_in_df:
    aggregated_df = aggregated_df.sort_values(['community_area', 'year'])
    for col in pop_cols_in_df:
        aggregated_df[col] = aggregated_df.groupby('community_area')[col].fillna(method='ffill').fillna(method='bfill')

# Add derived features
if 'homicide_count' in aggregated_df.columns and 'homicide_arrests' in aggregated_df.columns:
    aggregated_df['homicide_arrest_rate'] = aggregated_df.apply(
        lambda row: row['homicide_arrests'] / row['homicide_count'] if row['homicide_count'] > 0 else 0, axis=1
    )

if 'shooting_count' in aggregated_df.columns and 'shooting_arrests' in aggregated_df.columns:
    aggregated_df['shooting_arrest_rate'] = aggregated_df.apply(
        lambda row: row['shooting_arrests'] / row['shooting_count'] if row['shooting_count'] > 0 else 0, axis=1
    )

if 'pop_total' in aggregated_df.columns:
    aggregated_df['homicides_per_10k'] = aggregated_df.apply(
        lambda row: (row['homicide_count'] / row['pop_total'] * 10000) if row['pop_total'] > 0 else 0, axis=1
    )
    aggregated_df['shootings_per_10k'] = aggregated_df.apply(
        lambda row: (row['shooting_count'] / row['pop_total'] * 10000) if row['shooting_count'] > 0 else 0, axis=1
    )

print(f"✅ Aggregated dataset (2012+): {len(aggregated_df):,} rows × {len(aggregated_df.columns)} columns")

# ============================================================================
# STEP 3: CREATE COMPREHENSIVE UNAGGREGATED DATASET
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Creating Comprehensive Unaggregated Dataset")
print("=" * 80)

# Prepare each dataset for comprehensive merge
print("\n🔄 Preparing individual datasets for comprehensive merge...")

# Homicides - add dataset identifier
df_homicides_comp = df_homicides.copy()
df_homicides_comp['dataset'] = 'homicides'
df_homicides_comp['record_type'] = 'homicide'
df_homicides_comp['incident_id'] = df_homicides_comp['id']
df_homicides_comp['incident_date'] = pd.to_datetime(df_homicides_comp['date'], errors='coerce')
df_homicides_comp['primary_type'] = 'HOMICIDE'
df_homicides_comp['description'] = df_homicides_comp.get('description', 'HOMICIDE')
df_homicides_comp['location_description'] = df_homicides_comp.get('location_description', '')
df_homicides_comp['arrest_made'] = df_homicides_comp['arrest']
df_homicides_comp['domestic_incident'] = df_homicides_comp['domestic']

# Shootings - add dataset identifier
df_shootings_comp = df_shootings.copy()
df_shootings_comp['dataset'] = 'shootings'
df_shootings_comp['record_type'] = 'shooting'
df_shootings_comp['incident_id'] = df_shootings_comp['id']
df_shootings_comp['incident_date'] = pd.to_datetime(df_shootings_comp['date'], errors='coerce')
df_shootings_comp['primary_type'] = df_shootings_comp.get('primary_type', 'WEAPONS VIOLATION')
df_shootings_comp['description'] = df_shootings_comp.get('description', 'SHOOTING')
df_shootings_comp['location_description'] = df_shootings_comp.get('location_description', '')
df_shootings_comp['arrest_made'] = df_shootings_comp['arrest']
df_shootings_comp['domestic_incident'] = df_shootings_comp['domestic']

# Business Licenses - add dataset identifier
df_business_comp = df_business.copy()
df_business_comp['dataset'] = 'business'
df_business_comp['record_type'] = 'business_license'
df_business_comp['incident_id'] = df_business_comp['id']
df_business_comp['incident_date'] = pd.to_datetime(df_business_comp.get('license_term_start_date', ''), errors='coerce')
df_business_comp['primary_type'] = 'BUSINESS LICENSE'
df_business_comp['description'] = df_business_comp.get('license_description', '')
df_business_comp['location_description'] = df_business_comp.get('address', '')
df_business_comp['arrest_made'] = False
df_business_comp['domestic_incident'] = False

# Population - add dataset identifier (one record per community area per year)
df_population_comp = df_population.copy()
df_population_comp['dataset'] = 'population'
df_population_comp['record_type'] = 'population'
df_population_comp['incident_id'] = df_population_comp['community_area'].astype(str) + '_' + df_population_comp['year'].astype(str)
df_population_comp['incident_date'] = pd.to_datetime(df_population_comp['year'].astype(str) + '-01-01')
df_population_comp['primary_type'] = 'POPULATION'
df_population_comp['description'] = 'Population Count'
df_population_comp['location_description'] = ''
df_population_comp['arrest_made'] = False
df_population_comp['domestic_incident'] = False

# Sociodemographics - add dataset identifier (one record per community area)
df_socio_comp = df_socio.copy()
df_socio_comp['dataset'] = 'sociodemographics'
df_socio_comp['record_type'] = 'sociodemographic'
df_socio_comp['incident_id'] = df_socio_comp['community_area'].astype(str) + '_socio'
df_socio_comp['incident_date'] = pd.to_datetime('2010-01-01')  # Static data
df_socio_comp['year'] = 2010  # Static year
df_socio_comp['primary_type'] = 'SOCIODEMOGRAPHIC'
df_socio_comp['description'] = 'Sociodemographic Features'
df_socio_comp['location_description'] = ''
df_socio_comp['arrest_made'] = False
df_socio_comp['domestic_incident'] = False

# Select common columns for comprehensive dataset
common_cols = [
    'dataset', 'record_type', 'incident_id', 'incident_date', 'year', 'community_area',
    'primary_type', 'description', 'location_description', 'arrest_made', 'domestic_incident'
]

# Add all other columns from each dataset
all_cols = set()
for df in [df_homicides_comp, df_shootings_comp, df_business_comp, df_population_comp, df_socio_comp]:
    all_cols.update(df.columns)

# Create comprehensive dataset
print("\n🔗 Creating comprehensive unaggregated dataset...")
comprehensive_dfs = []

for name, df in [
    ('Homicides', df_homicides_comp),
    ('Shootings', df_shootings_comp), 
    ('Business', df_business_comp),
    ('Population', df_population_comp),
    ('Sociodemographics', df_socio_comp)
]:
    # Select available columns
    available_cols = [col for col in common_cols if col in df.columns]
    other_cols = [col for col in df.columns if col not in common_cols and col not in ['id', 'case_number']]
    selected_cols = available_cols + other_cols
    
    df_selected = df[selected_cols].copy()
    comprehensive_dfs.append(df_selected)
    print(f"   ✓ {name}: {len(df_selected):,} records")

# Combine all datasets
comprehensive_df = pd.concat(comprehensive_dfs, ignore_index=True, sort=False)

# Fill missing values in common columns
for col in common_cols:
    if col in comprehensive_df.columns:
        if comprehensive_df[col].dtype == 'object':
            comprehensive_df[col] = comprehensive_df[col].fillna('')
        else:
            comprehensive_df[col] = comprehensive_df[col].fillna(0)

print(f"✅ Comprehensive dataset: {len(comprehensive_df):,} rows × {len(comprehensive_df.columns)} columns")

# ============================================================================
# STEP 4: SAVE BOTH DATASETS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: Saving Both Datasets")
print("=" * 80)

# Save aggregated dataset (2012+)
aggregated_file = 'aggregated_chicago_dataset_2012plus.csv'
aggregated_df.to_csv(aggregated_file, index=False)
print(f"\n✅ Aggregated dataset (2012+) saved: {aggregated_file}")
print(f"   Size: {len(aggregated_df):,} rows × {len(aggregated_df.columns)} columns")

# Save comprehensive dataset
comprehensive_file = 'comprehensive_chicago_dataset.csv'
comprehensive_df.to_csv(comprehensive_file, index=False)
print(f"\n✅ Comprehensive dataset saved: {comprehensive_file}")
print(f"   Size: {len(comprehensive_df):,} rows × {len(comprehensive_df.columns)} columns")

# Create data dictionaries
print("\n📖 Creating data dictionaries...")

# Aggregated dictionary
agg_dict = pd.DataFrame({
    'Column_Name': aggregated_df.columns,
    'Data_Type': aggregated_df.dtypes.astype(str),
    'Non_Null_Count': aggregated_df.count(),
    'Null_Count': aggregated_df.isnull().sum(),
    'Null_Percentage': (aggregated_df.isnull().sum() / len(aggregated_df) * 100).round(2),
    'Unique_Values': [aggregated_df[col].nunique() for col in aggregated_df.columns],
    'Sample_Value': [str(aggregated_df[col].dropna().iloc[0]) if aggregated_df[col].notna().any() else '' for col in aggregated_df.columns]
})
agg_dict.to_csv('aggregated_dataset_dictionary.csv', index=False)

# Comprehensive dictionary
comp_dict = pd.DataFrame({
    'Column_Name': comprehensive_df.columns,
    'Data_Type': comprehensive_df.dtypes.astype(str),
    'Non_Null_Count': comprehensive_df.count(),
    'Null_Count': comprehensive_df.isnull().sum(),
    'Null_Percentage': (comprehensive_df.isnull().sum() / len(comprehensive_df) * 100).round(2),
    'Unique_Values': [comprehensive_df[col].nunique() for col in comprehensive_df.columns],
    'Sample_Value': [str(comprehensive_df[col].dropna().iloc[0]) if comprehensive_df[col].notna().any() else '' for col in comprehensive_df.columns]
})
comp_dict.to_csv('comprehensive_dataset_dictionary.csv', index=False)

print("✅ Data dictionaries saved")

# Summary statistics
print("\n📊 Dataset Summary:")
print(f"   Aggregated (2012+): {len(aggregated_df):,} rows")
print(f"   Comprehensive: {len(comprehensive_df):,} rows")
print(f"   Total unique incidents: {comprehensive_df['incident_id'].nunique():,}")

print("\n📊 Record types in comprehensive dataset:")
record_counts = comprehensive_df['record_type'].value_counts()
for record_type, count in record_counts.items():
    print(f"   {record_type}: {count:,} records")

print("\n📊 Datasets in comprehensive dataset:")
dataset_counts = comprehensive_df['dataset'].value_counts()
for dataset, count in dataset_counts.items():
    print(f"   {dataset}: {count:,} records")

print("\n" + "=" * 80)
print("🎉 BOTH DATASETS CREATED SUCCESSFULLY!")
print("=" * 80)
print("\n📁 Output Files:")
print("   1. aggregated_chicago_dataset_2012plus.csv - Aggregated data (2012+)")
print("   2. comprehensive_chicago_dataset.csv - All individual records")
print("   3. aggregated_dataset_dictionary.csv - Aggregated data dictionary")
print("   4. comprehensive_dataset_dictionary.csv - Comprehensive data dictionary")
print("\n🚀 Ready for detailed analysis!")
