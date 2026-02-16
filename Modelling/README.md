# Final Baseline Dataset

## Overview
This dataset is the finalized modelling dataset used for baseline regression.

## Target Definition
The modelling target is `homicides_per_10k_next_year`.

It is constructed by shifting `homicides_per_10k` forward by 1 year within each `community_area` using:

groupby("community_area").shift(-1)

The final year per community was removed due to missing target values.

Final row count: 900

---

## Leakage Prevention

The following variables were removed to prevent target leakage:

- homicide_count
- homicide_arrests
- homicide_arrest_rate
- homicide_domestic
- homicides_per_10k (original current-year rate)

All non-lagged shooting variables were also removed.

Only lagged (_lag1) shooting features are included.

---

## Lag Features

The following lag1 features were created using:

groupby("community_area").shift(1)

- shooting_count_lag1
- shootings_per_10k_lag1
- is_shooting_lag1
- shooting_arrests_lag1
- shooting_arrest_rate_lag1
- shooting_domestic_lag1

---

## Standardization

All continuous numeric predictors were standardized using z-score scaling.

Not standardized:
- homicides_per_10k_next_year (target)
- community_area
- year
- binary indicators