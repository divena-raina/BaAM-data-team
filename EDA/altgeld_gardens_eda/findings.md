# Altgeld Gardens (Community Area 54) - Key Findings

## Violence Patterns

**Homicides**
- Average: 4.6 per year
- Rate: 1.8 per 10k population
- 31% above citywide average
- Trend: Decreasing

**Shootings**
- Average: 118.1 per year
- Rate: 45.4 per 10k population  
- 29% above citywide average
- Trend: Decreasing

**Arrest Rates**
- Homicide arrests: 35.1%
- Shooting arrests: 35.8%

## Socioeconomic Conditions

**Hardship Index**: 98 (highest in Chicago)
- Citywide average: 49.2
- 99% above average

**Poverty Rate**: 56.5%
- Citywide average: 21.7%
- 160% above average

**Per Capita Income**: $8,201
- Citywide average: $25,858
- 68% below average

**Unemployment**: 34.6%
- Citywide average: 15.3%

**Education**: 27.5% without HS diploma

## Demographics

**Population**: Stable at ~26,010
- No significant change 2012-2025

**Business Activity**: Very limited
- Average 2.4 licenses per year
- Recent uptick in 2023-2025

## Outliers Detected

Using IQR method, found outliers in 6 variables:

1. **business_license_count** - 3 outliers (2023, 2024, 2025)
   - Recent increase in business activity
   
2. **unique_business_types** - 3 outliers (2023, 2024, 2025)
   - Diversification in recent years

3. **homicide_domestic** - 2 outliers (2016, 2023)

4. **homicide_count** - 1 outlier (2018)
   - Spike year worth investigating

5. **shooting_arrests** - 1 outlier (2019)

## Key Observations

1. **Extreme socioeconomic disadvantage**: Altgeld Gardens has the highest hardship index in Chicago with poverty and unemployment rates more than double the citywide average.

2. **Violence above average but improving**: Both homicide and shooting rates are elevated (29-31% above citywide) but show decreasing trends over the 14-year period.

3. **Limited economic activity**: Very few businesses operate in the area, though there's been a recent uptick starting in 2023.

4. **Stable population**: No significant population change suggests neither growth nor major out-migration.

5. **Arrest rate concerns**: Homicide arrest rate (35%) is below the citywide average, suggesting challenges in case clearance.

## Files Made

**Statistics**
- summary_statistics.csv
- temporal_trends.csv  
- altgeld_vs_citywide_comparison.csv
- correlation_matrix.csv
- strong_correlations.csv
- outlier_report.csv
- data_quality_report.csv

**Visualizations**
- crime_trends_over_time.png
- arrest_rates_over_time.png
- socioeconomic_trends.png
- population_and_business.png
- correlation_heatmap.png
- key_variables_distributions.png
- key_variables_boxplots.png



