# Sociodemographics EDA Folder

This folder contains a comprehensive exploratory data analysis (EDA) of the sociodemographics dataset.

## Contents

### Analysis Script
- **`sociodemographics_eda.py`** - Complete EDA script that generates all analysis and visualizations

### Summary Statistics
- **`summary_statistics.csv`** - Basic descriptive statistics for all variables
- **`detailed_statistics.csv`** - Comprehensive variable analysis including skewness, kurtosis, missing values

### Visualizations
- **`distribution_plots.png`** - Histogram distributions for all numeric variables
- **`box_plots_outliers.png`** - Box plots showing outlier detection
- **`correlation_heatmap.png`** - Correlation matrix heatmap
- **`outliers_visualization.png`** - Scatter plots highlighting outliers

### Analysis Results
- **`strong_correlations.csv`** - Strong correlation pairs (|r| > 0.5)
- **`outliers_analysis.csv`** - Detailed outlier analysis by variable
- **`eda_conclusions.csv`** - Summary metrics and data quality assessment

### Documentation
- **`EDA_SUMMARY_REPORT.md`** - Comprehensive summary report with key findings
- **`README.md`** - This file

## How to Use

### Run the Analysis
```bash
cd socio_data/EDA
python sociodemographics_eda.py
```

### View Results
1. **Quick Overview**: Read `EDA_SUMMARY_REPORT.md`
2. **Detailed Stats**: Open CSV files in Excel/Python
3. **Visualizations**: View PNG files for charts and plots

## Key Findings

- **Dataset Quality**: A+ (95/100) - High quality with minimal issues
- **Coverage**: 78 community areas with 6 sociodemographic variables
- **Strong Correlations**: 11 strong correlations (|r| > 0.5) identified
- **Outliers**: 13 outliers detected across 4 variables
- **Geographic Patterns**: Clear North vs South/West Side disparities

## Recommendations

1. **Feature Selection**: All variables suitable for modeling
2. **Standardization**: Recommended due to different scales
3. **Education Priority**: Strongest predictor across outcomes
4. **Hardship Index**: Effective composite measure

## Contact

For questions about this analysis, contact the Data Team.
