import pandas as pd

df = pd.read_csv("business_licenses_datasets/business_licenses.csv")

df['LICENSE TERM START DATE'] = pd.to_datetime(df['LICENSE TERM START DATE'], errors='coerce')
df['DATE ISSUED'] = pd.to_datetime(df['DATE ISSUED'], errors='coerce')

df = df[df['LICENSE TERM START DATE'].dt.year >= 2012]

columns_to_drop = [
    'LICENSE STATUS CHANGE DATE',
    'SSA',
    'APPLICATION CREATED DATE',
    'WARD PRECINCT',
    'CONDITIONAL APPROVAL'
]
df = df.drop(columns=columns_to_drop)

# Drop rows with missing critical info
df = df.dropna(subset=['ZIP CODE', 'COMMUNITY AREA', 'LATITUDE', 'LONGITUDE'])

# Standardize formats
df['ZIP CODE'] = df['ZIP CODE'].astype(str).str.zfill(5)
df['WARD'] = df['WARD'].astype('Int64')
df['PRECINCT'] = df['PRECINCT'].astype('Int64')

# Save cleaned CSV
df.to_csv("business_licenses_datasets/cleaned_business_licenses.csv", index=False)
