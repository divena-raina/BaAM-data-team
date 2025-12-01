
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

EXCLUDED_AREA = 46
EXCLUDED_NAME = "Greater Grand Crossing"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = REPO_ROOT / "merged data sets" / "aggregated" / "aggregated_chicago_dataset_2012plus.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


print(f"Loading data from {DATA_PATH}...")
df_full = pd.read_csv(DATA_PATH)

print(f"\nExcluding {EXCLUDED_NAME} (Community Area {EXCLUDED_AREA})...")
df = df_full[df_full['community_area'] != EXCLUDED_AREA].copy()
df_excluded = df_full[df_full['community_area'] == EXCLUDED_AREA].copy()

print(f"Full dataset: {len(df_full):,} rows")
print(f"After exclusion: {len(df):,} rows ({len(df_excluded):,} rows removed)")
print(f"Years covered: {df['year'].min()} to {df['year'].max()}")
print(f"Community areas: {df['community_area'].nunique()} (was 77)")


print("\n" + "="*80)
print("GENERATING SUMMARY STATISTICS")
print("="*80)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'community_area' in numeric_cols:
    numeric_cols.remove('community_area')
if 'year' in numeric_cols:
    numeric_cols.remove('year')

summary_stats = df[numeric_cols].describe().T
summary_stats['missing_values'] = df[numeric_cols].isnull().sum()
summary_stats['missing_pct'] = (summary_stats['missing_values'] / len(df)) * 100

summary_stats['median'] = df[numeric_cols].median()
summary_stats['skewness'] = df[numeric_cols].skew()
summary_stats['kurtosis'] = df[numeric_cols].kurtosis()

summary_stats = summary_stats[[
    'count', 'missing_values', 'missing_pct', 'mean', 'median', 
    'std', 'min', '25%', '50%', '75%', 'max', 'skewness', 'kurtosis'
]]

summary_stats.to_csv(OUTPUT_DIR / "summary_statistics_numeric.csv")
print(f"✓ Summary statistics saved")

missing = (
    df.isna()
    .sum()
    .rename("missing_count")
    .to_frame()
    .assign(missing_pct=lambda s: (s["missing_count"] / len(df)) * 100)
    .sort_values("missing_pct", ascending=False)
)
missing.to_csv(OUTPUT_DIR / "missing_values_summary.csv")
print(f"✓ Missing values summary saved")


print("\n" + "="*80)
print("CITYWIDE TRENDS (WITHOUT GRAND CROSSING)")
print("="*80)

trend_columns = [
    "homicide_count", "shooting_count", "homicide_arrests", "shooting_arrests",
    "homicide_domestic", "shooting_domestic", "pop_total",
    "business_license_count", "per_capita_income", "hardship_index"
]
existing_columns = [c for c in trend_columns if c in df.columns]

yearly = (
    df.groupby("year")[existing_columns]
    .sum(min_count=1)
    .sort_index()
)
yearly.to_csv(OUTPUT_DIR / "citywide_totals_by_year.csv")
print(f"✓ Citywide totals by year saved")

community_profile = (
    df.groupby("community_area")[existing_columns]
    .mean()
    .sort_values("homicide_count", ascending=False)
)
community_profile.to_csv(OUTPUT_DIR / "community_area_profile.csv")
print(f"✓ Community area profile saved")


print("\n" + "="*80)
print("COMPARING WITH GRAND CROSSING")
print("="*80)

comparison_metrics = [
    'homicide_count', 'shooting_count', 'homicide_arrests', 'shooting_arrests',
    'pop_total', 'business_license_count', 'pct_below_poverty', 
    'pct_unemployed', 'per_capita_income', 'hardship_index'
]

comparison_data = []
for col in comparison_metrics:
    if col in df.columns and col in df_excluded.columns:
        rest_mean = df[col].mean()
        gc_mean = df_excluded[col].mean()
        diff = gc_mean - rest_mean
        pct_diff = ((gc_mean - rest_mean) / rest_mean * 100) if rest_mean != 0 else 0
        
        comparison_data.append({
            'Metric': col.replace('_', ' ').title(),
            'Rest of Chicago': round(rest_mean, 2),
            'Grand Crossing': round(gc_mean, 2),
            'Difference': round(diff, 2),
            'Percent Difference': f"{pct_diff:+.1f}%"
        })

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv(OUTPUT_DIR / "grand_crossing_vs_rest_comparison.csv", index=False)
print(f"✓ Comparison analysis saved")


print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)

corr_vars = [col for col in [
    'homicide_count', 'shooting_count', 'homicides_per_10k', 'shootings_per_10k',
    'homicide_arrest_rate', 'shooting_arrest_rate', 'pop_total',
    'business_license_count', 'unique_business_types',
    'pct_housing_crowded', 'pct_below_poverty', 'pct_unemployed',
    'pct_without_hs_diploma', 'per_capita_income', 'hardship_index'
] if col in df.columns]

if len(corr_vars) >= 2:
    correlation_matrix = df[corr_vars].corr()
    correlation_matrix.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
    print(f"✓ Correlation matrix saved")
    
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
        print(f"✓ Strong correlations saved")


print("\n" + "="*80)
print("DETECTING OUTLIERS")
print("="*80)

outlier_report = []

for col in numeric_cols:
    if df[col].notna().sum() > 0:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            continue
            
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[
            (df[col] < lower_bound) | (df[col] > upper_bound)
        ]
        
        if len(outliers) > 0:
            outlier_report.append({
                'Variable': col,
                'Outlier Count': len(outliers),
                'Outlier %': round((len(outliers) / len(df)) * 100, 1),
                'Lower Bound': round(lower_bound, 2),
                'Upper Bound': round(upper_bound, 2),
                'Min Value': round(df[col].min(), 2),
                'Max Value': round(df[col].max(), 2)
            })

if outlier_report:
    outlier_df = pd.DataFrame(outlier_report).sort_values('Outlier Count', ascending=False)
    outlier_df.to_csv(OUTPUT_DIR / "outlier_report.csv", index=False)
    print(f"✓ Outlier report saved ({len(outlier_report)} variables with outliers)")


print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

distribution_columns = [
    "homicide_count", "shooting_count", "homicide_arrests", "shooting_arrests",
    "pop_total", "per_capita_income", "hardship_index",
    "homicides_per_10k", "shootings_per_10k"
]
distribution_columns = [c for c in distribution_columns if c in df.columns]

for column in distribution_columns:
    series = pd.to_numeric(df[column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        continue

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(series, bins=30, color="#1f77b4", edgecolor="black", alpha=0.75)
    ax.set_title(f"Distribution of {column.replace('_', ' ').title()}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    output_path = PLOTS_DIR / f"{column}_distribution.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

print(f"✓ Distribution plots saved")

cleaned_columns = []
boxplot_data = []
for column in distribution_columns:
    series = pd.to_numeric(df[column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        continue
    cleaned_columns.append(column)
    boxplot_data.append(series)

if cleaned_columns:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(
        boxplot_data,
        tick_labels=[c.replace('_', ' ')[:20] for c in cleaned_columns],
        patch_artist=True,
        boxprops=dict(facecolor="#aec7e8", color="#1f77b4"),
        medianprops=dict(color="#d62728"),
    )
    ax.set_title("Boxplots for Key Variables (Without Grand Crossing)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    boxplot_path = PLOTS_DIR / "key_variables_boxplot.png"
    fig.savefig(boxplot_path, dpi=150)
    plt.close(fig)
    print(f"✓ Boxplot saved")


if len(corr_vars) >= 2:
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, square=True, linewidths=1,
                cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap - Chicago Without Grand Crossing\n', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"✓ Correlation heatmap saved")
    plt.close()


yearly_city = df.groupby('year').agg({
    'homicide_count': 'sum',
    'shooting_count': 'sum',
    'pop_total': 'sum',
    'business_license_count': 'sum'
}).reset_index()

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Citywide Trends Over Time (Excluding Grand Crossing)', 
             fontsize=16, fontweight='bold')

axes[0, 0].plot(yearly_city['year'], yearly_city['homicide_count'], 
                marker='o', linewidth=2, color='#d62728')
axes[0, 0].set_title('Total Homicides by Year')
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Count')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(yearly_city['year'], yearly_city['shooting_count'], 
                marker='o', linewidth=2, color='#ff7f0e')
axes[0, 1].set_title('Total Shootings by Year')
axes[0, 1].set_xlabel('Year')
axes[0, 1].set_ylabel('Count')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(yearly_city['year'], yearly_city['pop_total'], 
                marker='o', linewidth=2, color='#2ca02c')
axes[1, 0].set_title('Total Population by Year')
axes[1, 0].set_xlabel('Year')
axes[1, 0].set_ylabel('Population')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(yearly_city['year'], yearly_city['business_license_count'], 
                marker='o', linewidth=2, color='#9467bd')
axes[1, 1].set_title('Total Business Licenses by Year')
axes[1, 1].set_xlabel('Year')
axes[1, 1].set_ylabel('Count')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "citywide_trends_over_time.png", dpi=300, bbox_inches='tight')
print(f"✓ Temporal trends plot saved")
plt.close()


top_areas = df.groupby('community_area').agg({
    'homicide_count': 'sum',
    'shooting_count': 'sum',
    'pop_total': 'mean'
}).sort_values('homicide_count', ascending=False).head(20)

fig, axes = plt.subplots(1, 2, figsize=(15, 8))
fig.suptitle('Top 20 Community Areas by Violence (Excluding Grand Crossing)', 
             fontsize=16, fontweight='bold')

axes[0].barh(range(len(top_areas)), top_areas['homicide_count'].values, color='#d62728')
axes[0].set_yticks(range(len(top_areas)))
axes[0].set_yticklabels([f"CA {int(ca)}" for ca in top_areas.index])
axes[0].set_xlabel('Total Homicides (2012-2025)')
axes[0].set_title('By Homicides')
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3, axis='x')

axes[1].barh(range(len(top_areas)), top_areas['shooting_count'].values, color='#ff7f0e')
axes[1].set_yticks(range(len(top_areas)))
axes[1].set_yticklabels([f"CA {int(ca)}" for ca in top_areas.index])
axes[1].set_xlabel('Total Shootings (2012-2025)')
axes[1].set_title('By Shootings')
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "top_community_areas.png", dpi=300, bbox_inches='tight')
print(f"✓ Community areas plot saved")
plt.close()


print("\n" + "="*80)
print("EDA COMPLETE")
print("="*80)
print(f"\nAll outputs saved to: {OUTPUT_DIR.resolve()}")
print(f"All plots saved to: {PLOTS_DIR.resolve()}")
print(f"\nDataset Summary:")
print(f"  - Total records: {len(df):,}")
print(f"  - Community areas: {df['community_area'].nunique()}")
print(f"  - Years: {df['year'].min()}-{df['year'].max()}")
print(f"  - Excluded: {EXCLUDED_NAME} (Area {EXCLUDED_AREA})")

