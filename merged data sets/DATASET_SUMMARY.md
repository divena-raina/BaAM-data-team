# Chicago Crime & Business Analysis Datasets

## Overview
This repository contains **organized datasets** for different analytical purposes:

1. **Aggregated Dataset** (2012+ only) - Recent aggregated format  
2. **Comprehensive Dataset** - All individual records unaggregated
3. **Summary Statistics** - Annual city-wide totals

**Organization**: All processed datasets are organized in the `merged data sets/` folder with subdirectories for easy navigation.

---

## Dataset Comparison

| Dataset | Rows | Columns | Size | Purpose |
|---------|------|---------|------|---------|
| **Summary** | 25 | 5 | 860 B | Annual city-wide totals |
| **Aggregated** | 1,091 | 27 | 205 KB | Recent analysis (2012+) |
| **Comprehensive** | 281,753 | 100 | 116 MB | Individual record analysis |

---

## 1. Summary Statistics (`summary_statistics_by_year.csv`)

**Purpose**: Quick annual overview across all of Chicago
- **Time Period**: 2001-2025 (25 years)
- **Format**: City-wide totals only
- **Records**: 25 (one per year)

**Key Features**:
- Annual totals for homicides, shootings, business licenses, population
- No geographic breakdown
- Quick trend overview

**Use Cases**:
- High-level trend analysis
- Annual reporting
- Dashboard summaries
- Quick city-wide statistics

---

## 2. Aggregated Dataset (`aggregated dataset/aggregated_chicago_dataset_2012plus.csv`)

**Purpose**: Focused analysis on recent data (2012+)
- **Time Period**: 2012-2025 (14 years)
- **Format**: Aggregated by community area × year
- **Records**: 1,091 (77 areas × 14 years)

**Key Features**:
- Same structure as master dataset
- Filtered to 2012+ only
- Includes shooting data (which starts in 2012)
- More recent business activity

**Use Cases**:
- Recent crime trends analysis
- Modern predictive modeling
- Resource allocation for current conditions

---

## 3. Comprehensive Dataset (`comprehensive/comprehensive_chicago_dataset.csv`)

**Purpose**: Individual record analysis across all datasets
- **Time Period**: 2001-2025 (varies by dataset)
- **Format**: Individual records with dataset identifiers
- **Records**: 281,753 individual incidents/records

**Record Types**:
- **Shootings**: 213,268 records (2012+)
- **Business Licenses**: 54,989 records
- **Homicides**: 13,359 records (2001+)
- **Sociodemographics**: 78 records (static)
- **Population**: 59 records (by year/area)

**Key Features**:
- Each record is a unique incident/entity
- Dataset identifier for filtering
- Record type classification
- All original fields preserved
- Case-by-case analysis capability

**Use Cases**:
- Individual incident analysis
- Pattern recognition in specific crimes
- Business location analysis
- Detailed geographic analysis
- Machine learning on individual records

---

## 📁 File Structure

```
BaAM-data-team/
├── merged data sets/
│   ├── aggregated dataset/
│   │   ├── aggregated_chicago_dataset_2012plus.csv      # Aggregated (2012+)
│   │   └── aggregated_dataset_dictionary.csv             # Aggregated dictionary
│   ├── comprehensive/
│   │   ├── comprehensive_chicago_dataset.csv             # Comprehensive (all records)
│   │   └── comprehensive_dataset_dictionary.csv          # Comprehensive dictionary
│   ├── summary_statistics_by_year.csv                   # Annual summaries
│   └── DATASET_SUMMARY.md                               # This documentation
```

---

## Data Quality Summary

### Summary Statistics
- **Complete coverage**: 2001-2025 (25 years)
- **City-wide totals**: All of Chicago
- **Simple format**: Easy to use

### Aggregated Dataset (2012+)
- **No missing values**: 100% complete
- **No duplicates**: Unique by (community_area, year)
- **Recent focus**: 2012-2025 only
- **Shooting data included**: Full coverage

### Comprehensive Dataset
- **All individual records**: 281,753 total
- **Dataset identifiers**: Easy filtering
- **Record type classification**: Clear categorization
- **Original fields preserved**: Maximum detail

---

## Usage Examples

### For Time Series Analysis
```python
# Use Aggregated dataset
df = pd.read_csv('aggregated dataset/aggregated_chicago_dataset_2012plus.csv')
```

### For Individual Record Analysis
```python
# Use Comprehensive dataset
df = pd.read_csv('comprehensive/comprehensive_chicago_dataset.csv')

# Filter by dataset
homicides = df[df['dataset'] == 'homicides']
shootings = df[df['dataset'] == 'shootings']
business = df[df['dataset'] == 'business']

# Filter by record type
crimes = df[df['record_type'].isin(['homicide', 'shooting'])]
```

### For Geographic Analysis
```python
# All datasets have community_area column
df = pd.read_csv('comprehensive/comprehensive_chicago_dataset.csv')
by_area = df.groupby('community_area').size()
```

### For Annual Overview
```python
# Use Summary Statistics
df = pd.read_csv('summary_statistics_by_year.csv')
```

---