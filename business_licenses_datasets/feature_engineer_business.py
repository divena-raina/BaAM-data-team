licenses_path   = "business_licenses_datasets/cleaned_business_licenses.csv"
population_path = "business_licenses_datasets/otherdatasets/Chicago_Population_Counts_Cleaned.csv"
socio_path      = "business_licenses_datasets/otherdatasets/sociodemographic_features_by_area.csv"
outdir          = "business_licenses_datasets"

import os, pandas as pd, numpy as np
os.makedirs(outdir, exist_ok=True)

def norm_ca(s):
    num = pd.to_numeric(s, errors="coerce")
    out = np.where(num.notna(), num.astype("Int64").astype(str), s.astype(str).str.strip())
    return pd.Series(out, index=s.index).astype(str)
def to_year(s): return pd.to_datetime(s, errors="coerce").dt.year
def any_kw(t, kws):
    t = str(t).lower()
    return any(k in t for k in kws)

lic = pd.read_csv(licenses_path)
try:
    pop = pd.read_csv(population_path)
except Exception:
    pop = None
socio = pd.read_csv(socio_path) if socio_path and os.path.exists(socio_path) else None

if "COMMUNITY AREA" in lic.columns: lic = lic.rename(columns={"COMMUNITY AREA":"community_area"})
elif "community_area" not in lic.columns: raise ValueError("Missing COMMUNITY AREA in licenses.")
issue_col  = "DATE ISSUED" if "DATE ISSUED" in lic.columns else None
expire_col = "LICENSE TERM EXPIRATION DATE" if "LICENSE TERM EXPIRATION DATE" in lic.columns else None
if issue_col is None: raise ValueError("Missing DATE ISSUED in licenses.")
if expire_col is None:
    lic["LICENSE TERM EXPIRATION DATE"] = pd.NaT
    expire_col = "LICENSE TERM EXPIRATION DATE"

if pop is not None:
    if "community_area_number" in pop.columns: pop = pop.rename(columns={"community_area_number":"community_area"})
    elif "community" in pop.columns: pop = pop.rename(columns={"community":"community_area"})
    if "Year" in pop.columns and "year" not in pop.columns: pop = pop.rename(columns={"Year":"year"})
    elif "year" not in pop.columns:
        c = [c for c in pop.columns if c.lower()=="year"]
        if c: pop = pop.rename(columns={c[0]:"year"})
        else: pop = None
    if pop is not None:
        if "Population - Total" in pop.columns and "population" not in pop.columns: pop = pop.rename(columns={"Population - Total":"population"})
        elif "population" not in pop.columns:
            c = [c for c in pop.columns if "population" in c.lower()]
            if c: pop = pop.rename(columns={c[0]:"population"})
            else: pop = None

if socio is not None and "Community Area" in socio.columns and "community_area" not in socio.columns:
    socio = socio.rename(columns={"Community Area":"community_area"})

if "community_area" in lic.columns:  lic["community_area"]  = norm_ca(lic["community_area"])
if pop is not None and "community_area" in pop.columns:    pop["community_area"]    = norm_ca(pop["community_area"])
if socio is not None and "community_area" in socio.columns: socio["community_area"] = norm_ca(socio["community_area"])

lic["issue_year"]  = to_year(lic[issue_col])
lic["expire_year"] = to_year(lic[expire_col])

years_lic = sorted(pd.to_numeric(lic["issue_year"], errors="coerce").dropna().unique())

if pop is not None:
    years_pop = sorted(pd.to_numeric(pop["year"], errors="coerce").dropna().unique())
    years = sorted(set(years_lic) & set(years_pop))
    if len(years) == 0:
        years = years_lic
        pop = None
else:
    years = years_lic

if len(years) == 0:
    raise ValueError("No valid years detected from input data.")


frames = []
for y in years:
    active = (lic["issue_year"] <= y) & (lic["expire_year"].isna() | (lic["expire_year"] >= y))
    lic_y = lic.loc[active].copy()
    g = lic_y.groupby("community_area", dropna=False)
    active_count = g.size().rename("active_count")
    desc_col = "LICENSE DESCRIPTION" if "LICENSE DESCRIPTION" in lic.columns else None
    apptype_col = "APPLICATION TYPE" if "APPLICATION TYPE" in lic.columns else None
    if desc_col and apptype_col: text = lic_y[desc_col].astype(str) + " " + lic_y[apptype_col].astype(str)
    elif desc_col: text = lic_y[desc_col].astype(str)
    elif apptype_col: text = lic_y[apptype_col].astype(str)
    else: text = pd.Series([""]*len(lic_y), index=lic_y.index)
    lic_y = lic_y.assign(
        nightlife_flag=[int(any_kw(x, ["bar","tavern","night","club","lounge","liquor","late-hour","late hour"])) for x in text],
        regulated_flag=[int(any_kw(x, ["liquor","tobacco","cigar","cigarette","firearm","gun","ammo","ammunition"])) for x in text]
    )
    g2 = lic_y.groupby("community_area", dropna=False)
    nightlife_share = g2["nightlife_flag"].mean().rename("nightlife_share")
    regulated_share = g2["regulated_flag"].mean().rename("regulated_share")
    new_count = lic.loc[lic["issue_year"] == y].groupby("community_area", dropna=False).size().rename("new_count")
    close_count = lic.loc[lic["expire_year"] == y].groupby("community_area", dropna=False).size().rename("closures_count")
    dfy = pd.concat([active_count, new_count, close_count, nightlife_share, regulated_share], axis=1).reset_index()
    dfy["year"] = y
    frames.append(dfy)

feat = pd.concat(frames, ignore_index=True).fillna(np.nan)

if pop is not None:
    feat = feat.merge(pop[["community_area","year","population"]], on=["community_area","year"], how="left")
    for s,d in [("active_count","active_businesses_per_1k"),("new_count","new_licenses_per_1k"),("closures_count","closures_per_1k")]:
        feat[d] = (feat[s] / feat["population"]) * 1000
else:
    feat["population"] = np.nan
    feat["active_businesses_per_1k"] = np.nan
    feat["new_licenses_per_1k"] = np.nan
    feat["closures_per_1k"] = np.nan

if socio is not None:
    if "year" in socio.columns: feat = feat.merge(socio, on=["community_area","year"], how="left")
    else: feat = feat.merge(socio, on="community_area", how="left")

yvals = pd.to_numeric(feat["year"], errors="coerce")
if yvals.notna().any():
    latest_year = int(yvals.max())
    latest = feat.loc[yvals == latest_year].drop(columns=["year"]).reset_index(drop=True)
else:
    latest_year = None
    latest = feat.drop(columns=["year"], errors="ignore").reset_index(drop=True)

p1 = os.path.join(outdir, "business_features_community_year.csv")
p2 = os.path.join(outdir, "business_features_latest_year.csv")
feat.to_csv(p1, index=False)
latest.to_csv(p2, index=False)
print("Wrote:", p1)
print("Wrote:", p2)
print("Latest year:", latest_year)

import pandas as pd, os
df_dict = pd.DataFrame([
    ("active_businesses_per_1k", "Active licenses per 1,000 residents in year Y (if population available).", "float"),
    ("new_licenses_per_1k",      "New licenses per 1,000 residents in year Y (if population available).",    "float"),
    ("closures_per_1k",          "License closures per 1,000 residents in year Y (if population available).","float"),
    ("nightlife_share",          "Share of ACTIVE licenses with nightlife keywords.",                         "float (0-1)"),
    ("regulated_share",          "Share of ACTIVE licenses with regulated-goods keywords.",                   "float (0-1)"),
    ("active_count",             "Raw number of ACTIVE licenses in year Y.",                                  "int"),
    ("new_count",                "Raw number of NEW licenses issued in year Y.",                              "int"),
    ("closures_count",           "Raw number of licenses that expired in year Y.",                            "int"),
], columns=["feature_name","description","data_type"])
df_dict.to_csv(os.path.join(outdir, "feature_dictionary_business.csv"), index=False)
print("Wrote:", os.path.join(outdir, "feature_dictionary_business.csv"))
