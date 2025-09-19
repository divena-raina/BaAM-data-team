# Chicago Shooting Crimes Dataset - Filtering and Cleaning Documentation

## Processing Information
- **Source file**: Crimes_-_2001_to_Present_20250918.csv
- **Original records**: 8,402,707
- **Processing date**: 2025-09-18 20:26:23
- **Year filter**: 2012 and onwards only

## Filtering Criteria

### Primary Weapon Keywords (in Description field)
- **HANDGUN**: 372,750 records
- **FIREARM**: Found in various combinations
- **RIFLE**: Found in air rifle and other contexts

### Discharge Keywords (in Description field)
- **RECKLESS FIREARM DISCHARGE**: 15,301 records
- **FIREARM DISCHARGE**: Broader discharge patterns
- **DISCHARGE**: Any discharge variations

*Note: The specified "DISCHARGED FIREARM" was not found in the dataset, but "RECKLESS FIREARM DISCHARGE" captures similar incidents.*

### Additional Shooting Identifiers
- **GUN**: Includes gun offender registrations and other gun-related crimes

## Filtering Results
- **Records with weapon keywords**: 372,750
- **Records with discharge keywords**: 15,301
- **Records with additional keywords**: 344,538
- **Total shooting-related records**: 378,124
- **After year filter (2012+)**: 213,268
- **Final cleaned records**: 213,268

## Data Cleaning Steps

### 1. Date Standardization
- Converted 'Date' and 'Updated On' to datetime format
- Created standardized date field (YYYY-MM-DD format)
- Added year, month, and day_of_week fields
- All dates successfully converted: 213,268 records

### 2. Numeric Field Standardization
- Standardized: Beat, District, Ward, Community Area, coordinates, Year
- Invalid values converted to NaN for proper handling

### 3. Categorical Field Standardization
- Trimmed whitespace from Primary Type, Description, Location Description
- Converted to consistent UPPERCASE format for standardization

### 4. Geography Validation
- **Community Areas**: Validated against Chicago's 77 community areas
  - Valid (1-77): 213,266 records
  - Invalid: 0 records
- **Wards**: Validated against Chicago's 50 wards
  - Valid (1-50): 213,266 records

### 5. Data Quality Filtering
- Removed records missing critical data (date, description, primary type, location)
- **Records removed**: 0
- **Records retained**: 213,268

## Final Dataset Characteristics

### Dataset Shape
- **Rows**: 213,268
- **Columns**: 26

### Time Coverage
- **Date range**: 2012-01-01 00:00:00 to 2025-09-10 00:00:00
- **Years**: 2012-2025

### Primary Crime Types
Primary Type
WEAPONS VIOLATION          76956
ROBBERY                    56636
ASSAULT                    46765
BATTERY                    26657
OTHER OFFENSE               5323
CRIM SEXUAL ASSAULT          483
CRIMINAL SEXUAL ASSAULT      445
NON-CRIMINAL                   3

### Top 15 Crime Descriptions
Description
AGGRAVATED - HANDGUN                  38195
AGGRAVATED: HANDGUN                   32874
ARMED: HANDGUN                        31371
UNLAWFUL POSSESSION - HANDGUN         28758
UNLAWFUL POSS OF HANDGUN              24034
ARMED - HANDGUN                       19632
RECKLESS FIREARM DISCHARGE            14566
ATTEMPT: ARMED-HANDGUN                 2683
UNLAWFUL USE HANDGUN                   2682
ATTEMPT ARMED - HANDGUN                1882
UNLAWFUL USE - OTHER FIREARM           1778
UNLAWFUL USE - HANDGUN                 1746
GUN OFFENDER: ANNUAL REGISTRATION      1694
GUN OFFENDER - ANNUAL REGISTRATION     1530
AGGRAVATED - OTHER FIREARM              985

### Geographic Coverage
- **Records with Community Area**: 213,266 (100.0%)
- **Records with Ward**: 213,266 (100.0%)
- **Records with Lat/Lon**: 212,879 (99.8%)

## Key Fields for Analysis

### Geographic Identifiers
- **Community Area**: For joining with demographic/socioeconomic data
- **Ward**: Political boundaries
- **District/Beat**: Police administrative areas
- **Latitude/Longitude**: Precise location coordinates

### Temporal Fields
- **date_standardized**: YYYY-MM-DD format
- **year_standardized**: Numeric year
- **month**: Month number (1-12)
- **day_of_week**: Day name (Monday, Tuesday, etc.)

### Crime Classification
- **Primary Type**: Main crime category
- **Description**: Specific crime description with weapon details
- **Location Description**: Type of location where crime occurred

## Data Quality Notes
- All records have valid dates and crime descriptions
- Geographic data coverage is high (>90% have coordinates or community area)
- Some records may have missing Ward or Community Area data
- All categorical fields have been standardized to uppercase
- Numeric fields have been properly typed with invalid values as NaN

## Usage Recommendations
- Use Community Area for demographic analysis and mapping
- Use date_standardized for time series analysis
- Filter by Primary Type for specific crime category analysis
- Use Description field for detailed weapon-specific filtering
- Geographic coordinates enable precise mapping and spatial analysis
