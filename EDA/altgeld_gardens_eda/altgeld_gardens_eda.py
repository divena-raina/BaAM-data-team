
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

COMMUNITY_AREA_ID = 54
COMMUNITY_AREA_NAME = "Altgeld Gardens"
INPUT_FILE = "../merged data sets/aggregated/aggregated_chicago_dataset_2012plus.csv"
OUTPUT_DIR = Path("altgeld_gardens_eda")
OUTPUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


print(f"Loading data from {INPUT_FILE}...")
df_full = pd.read_csv(INPUT_FILE)

# filtering for altgeld
df_altgeld = df_full[df_full['community_area'] == COMMUNITY_AREA_ID].copy()
df_altgeld = df_altgeld.sort_values('year')

# citywide data
df_city = df_full.groupby('year').agg({
    'homicide_count': 'sum',
    'shooting_count': 'sum',
    'homicide_arrests': 'sum',
    'shooting_arrests': 'sum',
    'pop_total': 'sum',
    'business_license_count': 'sum',
    'homicide_arrest_rate': 'mean',
    'shooting_arrest_rate': 'mean',
    'pct_housing_crowded': 'mean',
    'pct_below_poverty': 'mean',
    'pct_unemployed': 'mean',
    'pct_without_hs_diploma': 'mean',
    'per_capita_income': 'mean',
    'hardship_index': 'mean'
}).reset_index()

df_city['homicides_per_10k'] = (df_city['homicide_count'] / df_city['pop_total']) * 10000
df_city['shootings_per_10k'] = (df_city['shooting_count'] / df_city['pop_total']) * 10000

print(f"\nAltgeld Gardens records found: {len(df_altgeld)}")
print(f"Years covered: {df_altgeld['year'].min()} to {df_altgeld['year'].max()}")


print("\n" + "="*80)
print("GENERATING SUMMARY STATISTICS")
print("="*80)

numeric_cols = df_altgeld.select_dtypes(include=[np.number]).columns.tolist()
if 'community_area' in numeric_cols:
    numeric_cols.remove('community_area')
if 'year' in numeric_cols:
    numeric_cols.remove('year')

summary_stats = df_altgeld[numeric_cols].describe().T
summary_stats['missing_values'] = df_altgeld[numeric_cols].isnull().sum()
summary_stats['missing_pct'] = (summary_stats['missing_values'] / len(df_altgeld)) * 100

summary_stats['median'] = df_altgeld[numeric_cols].median()
summary_stats['skewness'] = df_altgeld[numeric_cols].skew()
summary_stats['kurtosis'] = df_altgeld[numeric_cols].kurtosis()

summary_stats = summary_stats[[
    'count', 'missing_values', 'missing_pct', 'mean', 'median', 
    'std', 'min', '25%', '50%', '75%', 'max', 'skewness', 'kurtosis'
]]

summary_stats.to_csv(OUTPUT_DIR / "summary_statistics.csv")
print(f"✓ Summary statistics saved to {OUTPUT_DIR / 'summary_statistics.csv'}")


quality_report = pd.DataFrame({
    'Metric': [
        'Total Records',
        'Years Covered',
        'Total Columns',
        'Numeric Columns',
        'Complete Records (no nulls)',
        'Completeness %',
        'Duplicate Records'
    ],
    'Value': [
        len(df_altgeld),
        f"{df_altgeld['year'].min()}-{df_altgeld['year'].max()}",
        len(df_altgeld.columns),
        len(numeric_cols),
        df_altgeld[numeric_cols].dropna().shape[0],
        f"{(df_altgeld[numeric_cols].dropna().shape[0] / len(df_altgeld)) * 100:.1f}%",
        df_altgeld.duplicated().sum()
    ]
})
quality_report.to_csv(OUTPUT_DIR / "data_quality_report.csv", index=False)



print("\n" + "="*80)
print("ANALYZING TEMPORAL TRENDS")
print("="*80)

# Key metrics over time
trend_metrics = [
    'homicide_count', 'shooting_count', 'homicides_per_10k', 'shootings_per_10k',
    'homicide_arrest_rate', 'shooting_arrest_rate', 'pop_total', 
    'business_license_count', 'hardship_index', 'pct_below_poverty'
]

temporal_summary = df_altgeld[['year'] + [col for col in trend_metrics if col in df_altgeld.columns]].copy()
temporal_summary.to_csv(OUTPUT_DIR / "temporal_trends.csv", index=False)
print(f"✓ Temporal trends saved to {OUTPUT_DIR / 'temporal_trends.csv'}")



print("\n" + "="*80)
print("COMPARING WITH CITYWIDE AVERAGES")
print("="*80)


comparison_metrics = {
    'homicides_per_10k': 'Homicides per 10k',
    'shootings_per_10k': 'Shootings per 10k',
    'homicide_arrest_rate': 'Homicide Arrest Rate',
    'shooting_arrest_rate': 'Shooting Arrest Rate',
    'pct_below_poverty': 'Below Poverty %',
    'pct_unemployed': 'Unemployed %',
    'pct_without_hs_diploma': 'Without HS Diploma %',
    'per_capita_income': 'Per Capita Income',
    'hardship_index': 'Hardship Index'
}

comparison_data = []
for col, label in comparison_metrics.items():
    if col in df_altgeld.columns and col in df_city.columns:
        altgeld_mean = df_altgeld[col].mean()
        city_mean = df_city[col].mean()
        diff = altgeld_mean - city_mean
        pct_diff = ((altgeld_mean - city_mean) / city_mean * 100) if city_mean != 0 else 0
        
        comparison_data.append({
            'Metric': label,
            'Altgeld Gardens': round(altgeld_mean, 2),
            'Citywide Average': round(city_mean, 2),
            'Difference': round(diff, 2),
            'Percent Difference': f"{pct_diff:+.1f}%"
        })

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv(OUTPUT_DIR / "altgeld_vs_citywide_comparison.csv", index=False)
print(f"✓ Comparison analysis saved to {OUTPUT_DIR / 'altgeld_vs_citywide_comparison.csv'}")

# correlation analysis

print("\n" + "="*80)
print("PERFORMING CORRELATION ANALYSIS")
print("="*80)

corr_vars = [col for col in [
    'homicide_count', 'shooting_count', 'homicides_per_10k', 'shootings_per_10k',
    'homicide_arrest_rate', 'shooting_arrest_rate', 'pop_total',
    'business_license_count', 'unique_business_types',
    'pct_housing_crowded', 'pct_below_poverty', 'pct_unemployed',
    'pct_without_hs_diploma', 'per_capita_income', 'hardship_index'
] if col in df_altgeld.columns]

correlation_matrix = df_altgeld[corr_vars].corr()
correlation_matrix.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
print(f"✓ Correlation matrix saved to {OUTPUT_DIR / 'correlation_matrix.csv'}")

# strong correlations (|r| > 0.7)
strong_corr = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) > 0.7:
            strong_corr.append({
                'Variable 1': correlation_matrix.columns[i],
                'Variable 2': correlation_matrix.columns[j],
                'Correlation': round(corr_val, 3),
                'Strength': 'Very Strong' if abs(corr_val) > 0.9 else 'Strong'
            })

if strong_corr:
    strong_corr_df = pd.DataFrame(strong_corr).sort_values('Correlation', key=abs, ascending=False)
    strong_corr_df.to_csv(OUTPUT_DIR / "strong_correlations.csv", index=False)
    print(f"✓ Strong correlations saved to {OUTPUT_DIR / 'strong_correlations.csv'}")

# outliers

print("\n" + "="*80)
print("DETECTING OUTLIERS")
print("="*80)

outlier_report = []

for col in numeric_cols:
    if df_altgeld[col].notna().sum() > 0:
        Q1 = df_altgeld[col].quantile(0.25)
        Q3 = df_altgeld[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df_altgeld[
            (df_altgeld[col] < lower_bound) | (df_altgeld[col] > upper_bound)
        ]
        
        if len(outliers) > 0:
            outlier_report.append({
                'Variable': col,
                'Outlier Count': len(outliers),
                'Outlier %': round((len(outliers) / len(df_altgeld)) * 100, 1),
                'Lower Bound': round(lower_bound, 2),
                'Upper Bound': round(upper_bound, 2),
                'Outlier Years': ', '.join(map(str, sorted(outliers['year'].tolist())))
            })

outlier_df = pd.DataFrame(outlier_report).sort_values('Outlier Count', ascending=False)
outlier_df.to_csv(OUTPUT_DIR / "outlier_report.csv", index=False)
print(f"✓ Outlier report saved to {OUTPUT_DIR / 'outlier_report.csv'}")

# PLOTS

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# 1 - trends over time
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(f'{COMMUNITY_AREA_NAME} - Crime Trends Over Time', fontsize=16, fontweight='bold')

axes[0, 0].plot(df_altgeld['year'], df_altgeld['homicide_count'], 
                marker='o', linewidth=2, color='#d62728', label='Altgeld Gardens')
axes[0, 0].axhline(df_altgeld['homicide_count'].mean(), 
                   color='red', linestyle='--', alpha=0.5, label='Mean')
axes[0, 0].set_title('Homicide Count by Year')
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Count')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(df_altgeld['year'], df_altgeld['shooting_count'], 
                marker='o', linewidth=2, color='#ff7f0e', label='Altgeld Gardens')
axes[0, 1].axhline(df_altgeld['shooting_count'].mean(), 
                   color='orange', linestyle='--', alpha=0.5, label='Mean')
axes[0, 1].set_title('Shooting Count by Year')
axes[0, 1].set_xlabel('Year')
axes[0, 1].set_ylabel('Count')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(df_altgeld['year'], df_altgeld['homicides_per_10k'], 
                marker='o', linewidth=2, color='#d62728', label='Altgeld Gardens')
axes[1, 0].plot(df_city['year'], df_city['homicides_per_10k'], 
                marker='s', linewidth=2, color='#1f77b4', alpha=0.6, label='Citywide')
axes[1, 0].set_title('Homicides per 10k Population')
axes[1, 0].set_xlabel('Year')
axes[1, 0].set_ylabel('Rate per 10k')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(df_altgeld['year'], df_altgeld['shootings_per_10k'], 
                marker='o', linewidth=2, color='#ff7f0e', label='Altgeld Gardens')
axes[1, 1].plot(df_city['year'], df_city['shootings_per_10k'], 
                marker='s', linewidth=2, color='#1f77b4', alpha=0.6, label='Citywide')
axes[1, 1].set_title('Shootings per 10k Population')
axes[1, 1].set_xlabel('Year')
axes[1, 1].set_ylabel('Rate per 10k')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "crime_trends_over_time.png", dpi=300, bbox_inches='tight')
print(f"✓ Crime trends plot saved")
plt.close()

# 2 - arrest rates over time
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle(f'{COMMUNITY_AREA_NAME} - Arrest Rates Over Time', fontsize=16, fontweight='bold')

axes[0].plot(df_altgeld['year'], df_altgeld['homicide_arrest_rate'], 
             marker='o', linewidth=2, color='#2ca02c', label='Altgeld Gardens')
axes[0].plot(df_city['year'], df_city['homicide_arrest_rate'], 
             marker='s', linewidth=2, color='#1f77b4', alpha=0.6, label='Citywide')
axes[0].set_title('Homicide Arrest Rate')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Arrest Rate')
axes[0].set_ylim(0, 1)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(df_altgeld['year'], df_altgeld['shooting_arrest_rate'], 
             marker='o', linewidth=2, color='#9467bd', label='Altgeld Gardens')
axes[1].plot(df_city['year'], df_city['shooting_arrest_rate'], 
             marker='s', linewidth=2, color='#1f77b4', alpha=0.6, label='Citywide')
axes[1].set_title('Shooting Arrest Rate')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Arrest Rate')
axes[1].set_ylim(0, 1)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "arrest_rates_over_time.png", dpi=300, bbox_inches='tight')
print(f"✓ Arrest rates plot saved")
plt.close()

# 3 - socioeconomic signs over time
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(f'{COMMUNITY_AREA_NAME} - Socioeconomic Indicators Over Time', 
             fontsize=16, fontweight='bold')

axes[0, 0].plot(df_altgeld['year'], df_altgeld['pct_below_poverty'], 
                marker='o', linewidth=2, color='#e377c2', label='Altgeld Gardens')
axes[0, 0].plot(df_city['year'], df_city['pct_below_poverty'], 
                marker='s', linewidth=2, color='#1f77b4', alpha=0.6, label='Citywide')
axes[0, 0].set_title('Below Poverty Rate')
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Percentage')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(df_altgeld['year'], df_altgeld['pct_unemployed'], 
                marker='o', linewidth=2, color='#bcbd22', label='Altgeld Gardens')
axes[0, 1].plot(df_city['year'], df_city['pct_unemployed'], 
                marker='s', linewidth=2, color='#1f77b4', alpha=0.6, label='Citywide')
axes[0, 1].set_title('Unemployment Rate')
axes[0, 1].set_xlabel('Year')
axes[0, 1].set_ylabel('Percentage')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(df_altgeld['year'], df_altgeld['per_capita_income'], 
                marker='o', linewidth=2, color='#17becf', label='Altgeld Gardens')
axes[1, 0].plot(df_city['year'], df_city['per_capita_income'], 
                marker='s', linewidth=2, color='#1f77b4', alpha=0.6, label='Citywide')
axes[1, 0].set_title('Per Capita Income')
axes[1, 0].set_xlabel('Year')
axes[1, 0].set_ylabel('Income ($)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(df_altgeld['year'], df_altgeld['hardship_index'], 
                marker='o', linewidth=2, color='#8c564b', label='Altgeld Gardens')
axes[1, 1].plot(df_city['year'], df_city['hardship_index'], 
                marker='s', linewidth=2, color='#1f77b4', alpha=0.6, label='Citywide')
axes[1, 1].set_title('Hardship Index')
axes[1, 1].set_xlabel('Year')
axes[1, 1].set_ylabel('Index')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "socioeconomic_trends.png", dpi=300, bbox_inches='tight')
print(f"✓ Socioeconomic trends plot saved")
plt.close()

# 4 - population and business activity
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle(f'{COMMUNITY_AREA_NAME} - Population & Business Activity', 
             fontsize=16, fontweight='bold')

axes[0].plot(df_altgeld['year'], df_altgeld['pop_total'], 
             marker='o', linewidth=2, color='#7f7f7f')
axes[0].set_title('Total Population Over Time')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Population')
axes[0].grid(True, alpha=0.3)

axes[1].plot(df_altgeld['year'], df_altgeld['business_license_count'], 
             marker='o', linewidth=2, color='#bcbd22')
axes[1].set_title('Business License Count Over Time')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Count')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "population_and_business.png", dpi=300, bbox_inches='tight')
print(f"✓ Population and business activity plot saved")
plt.close()

# 5 - correlation heatmap
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
            cmap='coolwarm', center=0, square=True, linewidths=1,
            cbar_kws={"shrink": 0.8})
plt.title(f'{COMMUNITY_AREA_NAME} - Correlation Heatmap\n', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
print(f"✓ Correlation heatmap saved")
plt.close()

# 6 - distribution plots for key variables
key_vars = ['homicide_count', 'shooting_count', 'homicides_per_10k', 'shootings_per_10k',
            'hardship_index', 'pct_below_poverty']
key_vars = [v for v in key_vars if v in df_altgeld.columns]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(f'{COMMUNITY_AREA_NAME} - Distribution of Key Variables', 
             fontsize=16, fontweight='bold')

for idx, var in enumerate(key_vars):
    row = idx // 3
    col = idx % 3
    axes[row, col].hist(df_altgeld[var].dropna(), bins=10, 
                        color='steelblue', edgecolor='black', alpha=0.7)
    axes[row, col].axvline(df_altgeld[var].mean(), color='red', 
                           linestyle='--', linewidth=2, label='Mean')
    axes[row, col].axvline(df_altgeld[var].median(), color='green', 
                           linestyle='--', linewidth=2, label='Median')
    axes[row, col].set_title(var.replace('_', ' ').title())
    axes[row, col].set_xlabel('Value')
    axes[row, col].set_ylabel('Frequency')
    axes[row, col].legend()
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "key_variables_distributions.png", dpi=300, bbox_inches='tight')
print(f"✓ Distribution plots saved")
plt.close()

# 7 - box plots for outliers
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(f'{COMMUNITY_AREA_NAME} - Box Plots (Outlier Detection)', 
             fontsize=16, fontweight='bold')

for idx, var in enumerate(key_vars):
    row = idx // 3
    col = idx % 3
    axes[row, col].boxplot(df_altgeld[var].dropna(), vert=True)
    axes[row, col].set_title(var.replace('_', ' ').title())
    axes[row, col].set_ylabel('Value')
    axes[row, col].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "key_variables_boxplots.png", dpi=300, bbox_inches='tight')
print(f"✓ Box plots saved")
plt.close()


