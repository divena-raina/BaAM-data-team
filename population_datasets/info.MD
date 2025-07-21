## Cleaning External Data 3
### Record ID
- Record ID is now set as the index, rather than a seperate column (unique identifier purposes).

### Geography
- Filtered out citywide population information (can be grabbed by aggregation functions)
- Rewrote geography column to be Zip Code column
- Added community area field to reflect which zip codes are associated with each community area using GeoPandas
    - Picked the largest overlap to determine community area

### Year
- Filtered to only 2021 population data (handling missing values)

### Datatypes
- Set entire index (not incl. community and index) to be integer datatype


