# Shooting dataset cleaning summary

Cleaning actions performed:
- Loaded 62828 rows from raw dataset.
- Dropped columns: HOMICIDE_VICTIM_FIRST_NAME, HOMICIDE_VICTIM_MI, HOMICIDE_VICTIM_LAST_NAME, STATE_HOUSE_DISTRICT, STATE_SENATE_DISTRICT, STREET_OUTREACH_ORGANIZATION.
- Filtered to gunshot victims only: removed 4322 rows.
- Standardized case numbers, street blocks, primary classifications, and ZIP codes for consistent joins.
- Filtered out incidents prior to 2012: removed 15026 rows.
- Dropped 24 rows missing essential identifiers or dates.
- Final cleaned dataset saved to `cleaned_data/shooting dataset pt3/cleaned_shooting_dataset_pt3.csv` with 43456 rows.