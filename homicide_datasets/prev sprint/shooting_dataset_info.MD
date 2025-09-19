## Clean and Prepare Shooting Dataset
### Dropped Columns and Missing Values
- 'Year', 'Location', 'Updated On', 'Block', 'IUCR', 'FBI Code', 'Primary Type', 'Description', 'X Coordinate', 'Y Coordinate' dropped (provided no additional information)
- Dropped 3 rows where values were missing

### Year
- Filtered to only 2012 data and greater (only relevant data)

### Datatypes
- Set index to be ID (unique identifier)
- Set year to datetime format (proper formatting)

Final dataset shape: 1,723 rows x 11 cols
