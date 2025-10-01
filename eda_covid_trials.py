#!/usr/bin/env python3
"""
eda_covid_trials.py
Simple, safe EDA pipeline for the COVID-19 Clinical Trials CSV.

Usage:
    python eda_covid_trials.py path/to/covid_clinical_trials.csv
Optional:
    python eda_covid_trials.py path/to/csv --out results_folder
"""

import os
import argparse
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def load_data(path):
    """Load CSV into a DataFrame (safe read)."""
    print(f"[+] Loading data from: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"[+] Loaded shape: {df.shape}")
    return df


def initial_explore(df, outdir):
    """Print and save quick initial exploration (head, info, missing)."""
    print("\n[+] First five rows:")
    print(df.head().to_string(max_rows=5))
    print("\n[+] Data types and non-null counts:")
    print(df.info())
    missing = (df.isnull().mean() * 100).sort_values(ascending=False)
    missing_path = os.path.join(outdir, "missing_percentages.csv")
    missing.to_csv(missing_path, header=["missing_pct"])
    print(f"[+] Missing percentage per column saved to: {missing_path}")
    return missing


def _safe_drop(df, cols):
    for c in cols:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
            print(f"[+] Dropped column: {c}")


def extract_country_from_locations(loc):
    """Heuristic: if Locations is 'City, State, Country', return Country (last token)."""
    if pd.isna(loc) or str(loc).strip() == "":
        return np.nan
    parts = str(loc).split(",")
    return parts[-1].strip()


def clean_data(df):
    """Perform cleaning & basic feature engineering. Returns a cleaned copy."""
    df = df.copy()
    # normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Drop very-high-missing known columns if present
    _safe_drop(df, ["Study Documents", "Results First Posted"])

    # Deduplicate
    before = df.shape[0]
    df.drop_duplicates(inplace=True)
    after = df.shape[0]
    print(f"[+] Dropped {before - after} duplicate rows (if any).")

    # Extract Country from Locations
    if "Locations" in df.columns:
        df["Country"] = df["Locations"].apply(extract_country_from_locations)
        print("[+] Extracted Country from Locations into new column 'Country'.")
    else:
        df["Country"] = np.nan

    # Fill categorical missing with a clear token
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna(f"Missing {c}")

    # Parse date-like columns safely
    date_cols = [
        "Start Date",
        "Primary Completion Date",
        "Completion Date",
        "First Posted",
        "Last Update Posted",
    ]
    for dc in date_cols:
        if dc in df.columns:
            df[dc] = pd.to_datetime(df[dc], errors="coerce")
            print(f"[+] Parsed dates in column: {dc}")

    # Enrollment: numeric coercion and median imputation
    if "Enrollment" in df.columns:
        df["Enrollment"] = pd.to_numeric(df["Enrollment"], errors="coerce")
        med = df["Enrollment"].median(skipna=True)
        df["Enrollment"] = df["Enrollment"].fillna(med)
        print(f"[+] Filled missing Enrollment with median: {med}")

    return df


def save_clean(df, outdir):
    path = os.path.join(outdir, "cleaned_covid_clinical_trials.csv")
    df.to_csv(path, index=False)
    print(f"[+] Cleaned data saved to: {path}")


def plot_bar_series(series, title, outdir, top_n=20, horizontal=True):
    """Save a simple bar plot for a value_counts series."""
    s = series.dropna()
    if s.empty:
        print(f"[!] No data to plot for {title}")
        return
    s = s.iloc[:top_n]
    plt.figure(figsize=(10, max(4, 0.3 * len(s))))
    if horizontal:
        sns.barplot(x=s.values, y=s.index)
        plt.xlabel("Count")
    else:
        sns.barplot(x=s.index, y=s.values)
        plt.xticks(rotation=90)
        plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    fname = os.path.join(outdir, f"{title.lower().replace(' ', '_')}.png")
    plt.savefig(fname)
    plt.close()
    print(f"[+] Saved plot: {fname}")


def univariate_plots(df, outdir):
    """Generate a few univariate plots and save them."""
    if "Status" in df.columns:
        plot_bar_series(df["Status"].value_counts(), "Status Distribution", outdir)
    if "Phases" in df.columns:
        plot_bar_series(df["Phases"].value_counts(), "Phases Distribution", outdir)
    if "Gender" in df.columns:
        plot_bar_series(df["Gender"].value_counts(), "Gender Distribution", outdir)
    if "Country" in df.columns:
        plot_bar_series(df["Country"].value_counts(), "Top Countries", outdir, top_n=25)

    # Enrollment distribution (remove absurd extremes for visibility)
    if "Enrollment" in df.columns:
        enroll = df["Enrollment"].replace([np.inf, -np.inf], np.nan).dropna()
        # limit very large outliers (these will be excluded from the plot for clarity)
        enroll_vis = enroll[enroll <= 1_000_000]
        plt.figure(figsize=(10, 5))
        plt.hist(enroll_vis, bins=50)
        plt.title("Enrollment Distribution (values <= 1,000,000)")
        plt.xlabel("Enrollment")
        plt.ylabel("Frequency")
        f = os.path.join(outdir, "enrollment_hist.png")
        plt.tight_layout()
        plt.savefig(f)
        plt.close()
        print(f"[+] Saved plot: {f}")


def bivariate_analysis(df, outdir):
    """Create a few bivariate views: phases vs status and time series of starts."""
    # Phase vs Status
    if "Phases" in df.columns and "Status" in df.columns:
        ct = pd.crosstab(df["Phases"], df["Status"])
        csvp = os.path.join(outdir, "phases_vs_status.csv")
        ct.to_csv(csvp)
        print(f"[+] Saved cross-tab CSV: {csvp}")
        ax = ct.plot(kind="bar", stacked=True, figsize=(12, 6))
        ax.set_title("Phase vs Status (stacked)")
        plt.tight_layout()
        f = os.path.join(outdir, "phases_vs_status.png")
        plt.savefig(f)
        plt.close()
        print(f"[+] Saved plot: {f}")

    # Trials by start date (monthly)
    if "Start Date" in df.columns:
        s = df["Start Date"].dropna()
        if not s.empty:
            monthly = s.dt.to_period("M").value_counts().sort_index()
            monthly_ts = monthly.copy()
            monthly_ts.index = monthly_ts.index.to_timestamp()
            csvp = os.path.join(outdir, "trials_started_monthly.csv")
            monthly_ts.to_csv(csvp, header=["n_trials"])
            print(f"[+] Saved monthly trials CSV: {csvp}")
            plt.figure(figsize=(12, 5))
            monthly_ts.plot(kind="line")
            plt.title("Number of Trials Started Over Time (monthly)")
            plt.xlabel("Month")
            plt.ylabel("Number of trials")
            plt.tight_layout()
            f = os.path.join(outdir, "trials_started_time_series.png")
            plt.savefig(f)
            plt.close()
            print(f"[+] Saved plot: {f}")

    # Top conditions and their outcome measures (text aggregation)
    if "Conditions" in df.columns and "Outcome Measures" in df.columns:
        top_conditions = df["Conditions"].value_counts().nlargest(20).index
        cond_outcomes = (
            df[df["Conditions"].isin(top_conditions)]
            .groupby("Conditions")["Outcome Measures"]
            .agg(lambda x: " || ".join(pd.unique(x.astype(str))))
        )
        csvp = os.path.join(outdir, "top_conditions_outcomes.csv")
        cond_outcomes.to_csv(csvp, header=["Outcome Measures Aggregated"])
        print(f"[+] Saved aggregated outcomes for top conditions: {csvp}")


def make_output_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(__doc__),
    )
    p.add_argument("csv", help="Path to covid_clinical_trials.csv (the CSV file).")
    p.add_argument("--out", "-o", default="output", help="Output directory (default: ./output)")
    args = p.parse_args()

    make_output_dir(args.out)
    df = load_data(args.csv)
    _ = initial_explore(df, args.out)
    df_clean = clean_data(df)
    save_clean(df_clean, args.out)
    univariate_plots(df_clean, args.out)
    bivariate_analysis(df_clean, args.out)
    print("\n[+] All done. Check the output directory for cleaned CSV, plots and CSV summaries.")


if __name__ == "__main__":
    main()
