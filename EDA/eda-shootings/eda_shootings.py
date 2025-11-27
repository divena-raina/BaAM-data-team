
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = REPO_ROOT / "cleaned_data" / "shooting dataset pt3" / "cleaned_shooting_dataset_pt3.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


print(f"Loading data from {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)

df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
df['YEAR'] = df['DATE'].dt.year
df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce')
df['HOUR'] = pd.to_numeric(df['HOUR'], errors='coerce')

print(f"\nDataset loaded: {len(df):,} rows and {df.shape[1]} columns")
print(f"Date range: {df['DATE'].min()} to {df['DATE'].max()}")


print("\n" + "="*80)
print("GENERATING SUMMARY STATISTICS")
print("="*80)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

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
print(f"✓ Numeric summary statistics saved")

categorical_cols = ['SEX', 'RACE', 'LOCATION_DESCRIPTION', 'DAY_OF_WEEK', 
                   'VICTIMIZATION_PRIMARY', 'INCIDENT_PRIMARY']
cat_summary = []
for col in categorical_cols:
    if col in df.columns:
        value_counts = df[col].value_counts()
        cat_summary.append({
            "column": col,
            "unique_count": df[col].nunique(),
            "most_common": value_counts.index[0] if len(value_counts) > 0 else "N/A",
            "most_common_count": value_counts.iloc[0] if len(value_counts) > 0 else 0,
            "most_common_pct": (value_counts.iloc[0] / len(df)) * 100 if len(value_counts) > 0 else 0,
            "missing_count": df[col].isnull().sum(),
            "missing_pct": (df[col].isnull().sum() / len(df)) * 100
        })

cat_summary_df = pd.DataFrame(cat_summary)
cat_summary_df.to_csv(OUTPUT_DIR / "summary_statistics_categorical.csv", index=False)
print(f"✓ Categorical summary statistics saved")

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

quality_report = pd.DataFrame({
    'Metric': [
        'Total Records',
        'Years Covered',
        'Total Columns',
        'Numeric Columns',
        'Categorical Columns',
        'Complete Records (no nulls)',
        'Completeness %',
        'Duplicate Records'
    ],
    'Value': [
        len(df),
        f"{df['YEAR'].min():.0f}-{df['YEAR'].max():.0f}" if 'YEAR' in df.columns else "N/A",
        len(df.columns),
        len(numeric_cols),
        len([c for c in categorical_cols if c in df.columns]),
        df.dropna().shape[0],
        f"{(df.dropna().shape[0] / len(df)) * 100:.1f}%",
        df.duplicated().sum()
    ]
})
quality_report.to_csv(OUTPUT_DIR / "data_quality_report.csv", index=False)
print(f"✓ Data quality report saved")


print("\n" + "="*80)
print("TEMPORAL TRENDS ANALYSIS")
print("="*80)

yearly_trends = df.groupby('YEAR').agg({
    'UNIQUE_ID': 'count',
    'AGE': 'mean',
    'COMMUNITY_AREA': 'nunique',
    'WARD': 'nunique'
}).rename(columns={'UNIQUE_ID': 'incident_count'})
yearly_trends.to_csv(OUTPUT_DIR / "temporal_trends.csv")
print(f"✓ Temporal trends saved")

monthly_trends = df.groupby('MONTH')['UNIQUE_ID'].count().to_frame('incident_count')
monthly_trends.to_csv(OUTPUT_DIR / "monthly_patterns.csv")
print(f"✓ Monthly patterns saved")

dow_trends = df.groupby('DAY_OF_WEEK')['UNIQUE_ID'].count().to_frame('incident_count')
dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_trends = dow_trends.reindex([d for d in dow_order if d in dow_trends.index])
dow_trends.to_csv(OUTPUT_DIR / "day_of_week_patterns.csv")
print(f"✓ Day of week patterns saved")

if 'HOUR' in df.columns:
    hourly_trends = df.groupby('HOUR')['UNIQUE_ID'].count().to_frame('incident_count')
    hourly_trends.to_csv(OUTPUT_DIR / "hourly_patterns.csv")
    print(f"✓ Hourly patterns saved")


print("\n" + "="*80)
print("DEMOGRAPHIC ANALYSIS")
print("="*80)

if 'SEX' in df.columns:
    sex_dist = df['SEX'].value_counts().to_frame('count')
    sex_dist['percentage'] = (sex_dist['count'] / len(df)) * 100
    sex_dist.to_csv(OUTPUT_DIR / "victim_sex_distribution.csv")
    print(f"✓ Sex distribution saved")

if 'RACE' in df.columns:
    race_dist = df['RACE'].value_counts().to_frame('count')
    race_dist['percentage'] = (race_dist['count'] / len(df)) * 100
    race_dist.to_csv(OUTPUT_DIR / "victim_race_distribution.csv")
    print(f"✓ Race distribution saved")

if 'AGE' in df.columns:
    age_stats = df['AGE'].describe().to_frame('statistics')
    age_stats.to_csv(OUTPUT_DIR / "victim_age_statistics.csv")
    print(f"✓ Age statistics saved")


print("\n" + "="*80)
print("GEOGRAPHIC ANALYSIS")
print("="*80)

if 'COMMUNITY_AREA' in df.columns:
    community_counts = df['COMMUNITY_AREA'].value_counts().head(20).to_frame('incident_count')
    community_counts['percentage'] = (community_counts['incident_count'] / len(df)) * 100
    community_counts.to_csv(OUTPUT_DIR / "top_20_community_areas.csv")
    print(f"✓ Top 20 community areas saved")

if 'LOCATION_DESCRIPTION' in df.columns:
    location_counts = df['LOCATION_DESCRIPTION'].value_counts().head(20).to_frame('count')
    location_counts['percentage'] = (location_counts['count'] / len(df)) * 100
    location_counts.to_csv(OUTPUT_DIR / "top_20_location_types.csv")
    print(f"✓ Top 20 location types saved")


print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)

corr_vars = [col for col in ['AGE', 'HOUR', 'MONTH', 'COMMUNITY_AREA', 'WARD', 'DISTRICT', 
                              'LATITUDE', 'LONGITUDE', 'YEAR'] if col in df.columns]

corr_df = df[corr_vars].copy()
for col in corr_vars:
    corr_df[col] = pd.to_numeric(corr_df[col], errors='coerce')

if len(corr_vars) >= 2:
    correlation_matrix = corr_df.corr()
    correlation_matrix.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
    print(f"✓ Correlation matrix saved")
    
    strong_corr = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                strong_corr.append({
                    'Variable 1': correlation_matrix.columns[i],
                    'Variable 2': correlation_matrix.columns[j],
                    'Correlation': round(corr_val, 3),
                    'Strength': 'Very Strong' if abs(corr_val) > 0.7 else 'Strong'
                })
    
    if strong_corr:
        strong_corr_df = pd.DataFrame(strong_corr).sort_values('Correlation', key=abs, ascending=False)
        strong_corr_df.to_csv(OUTPUT_DIR / "strong_correlations.csv", index=False)
        print(f"✓ Strong correlations saved")


print("\n" + "="*80)
print("DETECTING OUTLIERS")
print("="*80)

outlier_report = []

outlier_cols = [c for c in ['AGE', 'HOUR', 'COMMUNITY_AREA'] if c in df.columns]

for col in outlier_cols:
    col_data = pd.to_numeric(df[col], errors='coerce')
    if col_data.notna().sum() > 0:
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[
            (col_data < lower_bound) | (col_data > upper_bound)
        ]
        
        if len(outliers) > 0:
            outlier_report.append({
                'Variable': col,
                'Outlier Count': len(outliers),
                'Outlier %': round((len(outliers) / len(df)) * 100, 1),
                'Lower Bound': round(lower_bound, 2),
                'Upper Bound': round(upper_bound, 2),
                'Min Value': round(col_data.min(), 2),
                'Max Value': round(col_data.max(), 2)
            })

if outlier_report:
    outlier_df = pd.DataFrame(outlier_report).sort_values('Outlier Count', ascending=False)
    outlier_df.to_csv(OUTPUT_DIR / "outlier_report.csv", index=False)
    print(f"✓ Outlier report saved")


print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Shooting Incidents - Temporal Trends', fontsize=16, fontweight='bold')

axes[0, 0].plot(yearly_trends.index, yearly_trends['incident_count'], 
                marker='o', linewidth=2, color='#d62728')
axes[0, 0].set_title('Incidents by Year')
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Count')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].bar(monthly_trends.index, monthly_trends['incident_count'], color='#ff7f0e')
axes[0, 1].set_title('Incidents by Month')
axes[0, 1].set_xlabel('Month')
axes[0, 1].set_ylabel('Count')
axes[0, 1].grid(True, alpha=0.3, axis='y')

if not dow_trends.empty:
    axes[1, 0].barh(range(len(dow_trends)), dow_trends['incident_count'].values, color='#2ca02c')
    axes[1, 0].set_yticks(range(len(dow_trends)))
    axes[1, 0].set_yticklabels(dow_trends.index)
    axes[1, 0].set_title('Incidents by Day of Week')
    axes[1, 0].set_xlabel('Count')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    axes[1, 0].invert_yaxis()

if 'HOUR' in df.columns:
    axes[1, 1].plot(hourly_trends.index, hourly_trends['incident_count'], 
                    marker='o', linewidth=2, color='#9467bd')
    axes[1, 1].set_title('Incidents by Hour of Day')
    axes[1, 1].set_xlabel('Hour')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_xticks(range(0, 24, 2))
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "temporal_trends.png", dpi=300, bbox_inches='tight')
print(f"✓ Temporal trends plot saved")
plt.close()


if 'LOCATION_DESCRIPTION' in df.columns:
    fig, ax = plt.subplots(figsize=(12, 8))
    location_counts_plot = df['LOCATION_DESCRIPTION'].value_counts().head(15)
    ax.barh(range(len(location_counts_plot)), location_counts_plot.values, color='#1f77b4')
    ax.set_yticks(range(len(location_counts_plot)))
    ax.set_yticklabels(location_counts_plot.index)
    ax.set_xlabel('Count')
    ax.set_title('Top 15 Location Types for Shooting Incidents', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "location_types_distribution.png", dpi=300, bbox_inches='tight')
    print(f"✓ Location types plot saved")
    plt.close()


if 'SEX' in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Victim Demographics', fontsize=16, fontweight='bold')
    
    sex_counts = df['SEX'].value_counts()
    axes[0].pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%', 
                startangle=90, colors=['#ff9999', '#66b3ff', '#99ff99'])
    axes[0].set_title('Distribution by Sex')
    
    if 'RACE' in df.columns:
        race_counts = df['RACE'].value_counts().head(8)
        axes[1].barh(range(len(race_counts)), race_counts.values, color='#ff7f0e')
        axes[1].set_yticks(range(len(race_counts)))
        axes[1].set_yticklabels(race_counts.index)
        axes[1].set_xlabel('Count')
        axes[1].set_title('Top 8 Races')
        axes[1].invert_yaxis()
        axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "victim_demographics.png", dpi=300, bbox_inches='tight')
    print(f"✓ Demographics plot saved")
    plt.close()


if 'AGE' in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Victim Age Distribution', fontsize=16, fontweight='bold')
    
    age_clean = df['AGE'].dropna()
    age_clean = age_clean[(age_clean >= 0) & (age_clean <= 100)]
    
    axes[0].hist(age_clean, bins=30, color='#2ca02c', edgecolor='black', alpha=0.7)
    axes[0].axvline(age_clean.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {age_clean.mean():.1f}')
    axes[0].axvline(age_clean.median(), color='blue', linestyle='--', linewidth=2, label=f'Median: {age_clean.median():.1f}')
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Age Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].boxplot(age_clean, vert=True)
    axes[1].set_ylabel('Age')
    axes[1].set_title('Age Box Plot')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "victim_age_distribution.png", dpi=300, bbox_inches='tight')
    print(f"✓ Age distribution plot saved")
    plt.close()


if 'COMMUNITY_AREA' in df.columns:
    fig, ax = plt.subplots(figsize=(12, 8))
    ca_counts = df['COMMUNITY_AREA'].value_counts().head(20)
    ax.barh(range(len(ca_counts)), ca_counts.values, color='#e377c2')
    ax.set_yticks(range(len(ca_counts)))
    ax.set_yticklabels([str(ca) for ca in ca_counts.index])
    ax.set_xlabel('Incident Count')
    ax.set_title('Top 20 Community Areas by Shooting Incidents', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "community_areas_distribution.png", dpi=300, bbox_inches='tight')
    print(f"✓ Community areas plot saved")
    plt.close()


if len(corr_vars) >= 2:
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, square=True, linewidths=1,
                cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap - Numeric Variables\n', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"✓ Correlation heatmap saved")
    plt.close()


if 'VICTIMIZATION_PRIMARY' in df.columns:
    fig, ax = plt.subplots(figsize=(12, 8))
    vic_counts = df['VICTIMIZATION_PRIMARY'].value_counts().head(15)
    ax.barh(range(len(vic_counts)), vic_counts.values, color='#bcbd22')
    ax.set_yticks(range(len(vic_counts)))
    ax.set_yticklabels([str(v)[:40] for v in vic_counts.index])
    ax.set_xlabel('Count')
    ax.set_title('Top 15 Victimization Types', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "victimization_types.png", dpi=300, bbox_inches='tight')
    print(f"✓ Victimization types plot saved")
    plt.close()


print("\n" + "="*80)
print("EDA COMPLETE")
print("="*80)
print(f"\nAll outputs saved to: {OUTPUT_DIR.resolve()}")
print(f"All plots saved to: {PLOTS_DIR.resolve()}")

