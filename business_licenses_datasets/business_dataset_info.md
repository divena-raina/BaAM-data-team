# Cleaned and Prepared Business Licenses Dataset

## Dropped Columns and Missing Values
- Dropped columns that provided no additional value or were mostly null:
  - `LICENSE STATUS CHANGE DATE`
  - `SSA`
  - `APPLICATION CREATED DATE`
  - `WARD PRECINCT`
  - `CONDITIONAL APPROVAL`
- Dropped rows where ZIP CODE, COMMUNITY AREA, LATITUDE, or LONGITUDE were missing

## Year
- Filtered dataset to only include records with `LICENSE TERM START DATE` from 2012 onward

## Datatypes
- Standardized:
  - `ZIP CODE` → string (preserved leading zeroes)
  - `WARD`, `PRECINCT` → Int64 for compatibility with nulls
  - `LICENSE TERM START DATE` and `DATE ISSUED` → datetime format

## Final Dataset Shape
- 49,591 rows × 32 columns

## Potential Non-Critical Issues
- Some non-essential columns still have missing values:
  - `DOING BUSINESS AS NAME`: 40 missing (legal name is usually present)
  - `WARD`, `PRECINCT`, `POLICE DISTRICT`: a few dozen missing
  - `COMMUNITY AREA NAME`, `NEIGHBORHOOD`: minor gaps
  - `BUSINESS ACTIVITY` fields: ~3,000 missing (not always applicable)
  - `PAYMENT DATE`, `LICENSE APPROVED FOR ISSUANCE`: occasionally missing, but not required for most use cases