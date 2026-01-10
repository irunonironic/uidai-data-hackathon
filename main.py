import pandas as pd
import glob
import matplotlib.pyplot as plt
import os
import numpy as np
# =============================
# CONFIG
# =============================

DEMOGRAPHIC_PATH = "data/demographic/*.csv"
ENROLMENT_PATH = "data/enrollment/*.csv"

OUTPUT_CHARTS = "outputs/charts"
OUTPUT_TABLES = "outputs/tables"

os.makedirs(OUTPUT_CHARTS, exist_ok=True)
os.makedirs(OUTPUT_TABLES, exist_ok=True)
os.makedirs(f"{OUTPUT_TABLES}/districts", exist_ok=True)
os.makedirs(f"{OUTPUT_TABLES}/pincodes", exist_ok=True)
os.makedirs(f"{OUTPUT_TABLES}/heatmaps", exist_ok=True)

VALID_STATES = [
    "Andaman & Nicobar Islands", "Andhra Pradesh", "Arunachal Pradesh",
    "Assam", "Bihar", "Chhattisgarh", "Delhi", "Goa", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Ladakh", "Lakshadweep", "Madhya Pradesh",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland",
    "Odisha", "Puducherry", "Punjab", "Rajasthan", "Sikkim",
    "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal",
    "Dadra and Nagar Haveli and Daman and Diu", "Jammu And Kashmir"
]

# =============================
# UTILITIES
# =============================

def clean_state(name):
    if pd.isna(name):
        return name

    name = name.lower().strip().replace("&", "and")
    name = " ".join(name.split())

    corrections = {
        "west bengal": "West Bengal",
        "westbengal": "West Bengal",
        "west bangal": "West Bengal",
        "tamilnadu": "Tamil Nadu",
        "rajastan": "Rajasthan",
        "orissa": "Odisha",
        "telengana": "Telangana",
        "andaman and nicobar islands": "Andaman & Nicobar Islands",
        "daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
        "dadra and nagar haveli": "Dadra and Nagar Haveli and Daman and Diu",
    }

    return corrections.get(name, name.title())


# =============================
# LOADERS
# =============================

def load_demographic_data():
    print("\n--- Loading Demographic Data ---")
    files = glob.glob(DEMOGRAPHIC_PATH)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    df["total_updates"] = df["demo_age_5_17"] + df["demo_age_17_"]
    df["state_clean"] = df["state"].apply(clean_state)

    before = len(df)
    df = df[df["state_clean"].isin(VALID_STATES)].copy()
    print("Removed invalid demographic rows:", before - len(df))

    return df


def load_enrolment_data():
    print("\n--- Loading Enrolment Data ---")
    files = glob.glob(ENROLMENT_PATH)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    df["total_enrollments"] = (
        df["age_0_5"] + df["age_5_17"] + df["age_18_greater"]
    )
    df["state_clean"] = df["state"].apply(clean_state)

    before = len(df)
    df = df[df["state_clean"].isin(VALID_STATES)].copy()
    print("Removed invalid enrolment rows:", before - len(df))

    return df


# =============================
# AGGREGATION
# =============================

def aggregate_state_month(df):
    print("\n--- Aggregation ---")

    state_month = (
        df.groupby(["state_clean", "year_month"])["total_updates"]
        .sum()
        .reset_index()
        .sort_values(["state_clean", "year_month"])
    )

    state_month["prev_updates"] = (
        state_month.groupby("state_clean")["total_updates"].shift(1)
    )

    state_month["growth_pct"] = (
        (state_month["total_updates"] - state_month["prev_updates"])
        / state_month["prev_updates"]
    ) * 100

    state_month["abs_increase"] = (
        state_month["total_updates"] - state_month["prev_updates"]
    )

    state_month["zscore"] = (
        (state_month["total_updates"] -
         state_month.groupby("state_clean")["total_updates"].transform("mean")) /
        state_month.groupby("state_clean")["total_updates"].transform("std")
    ).fillna(0)

    return state_month


# =============================
# ANOMALY DETECTION
# =============================

def detect_anomalies(state_month):
    print("\n--- Anomaly Detection ---")

    anomalies = state_month[
        (state_month["growth_pct"] > 100) &
        (state_month["abs_increase"] > 20000) &
        (state_month["zscore"] > 1.5)
    ].copy()

    anomalies.to_csv(f"{OUTPUT_TABLES}/anomalies.csv", index=False)
    print("Anomalies detected:", len(anomalies))

    return anomalies


# =============================
# HOTSPOT ANALYSIS
# =============================

def hotspot_analysis(df, anomalies):
    print("\n--- Hotspot Analysis ---")

    top_rows = anomalies.sort_values("growth_pct", ascending=False).head(5)

    for _, row in top_rows.iterrows():
        state = row["state_clean"]
        month = row["year_month"]

        subset = df[
            (df["state_clean"] == state) &
            (df["year_month"] == month)
        ]

        # Districts
        district_summary = (
            subset.groupby("district")["total_updates"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        district_file = f"{OUTPUT_TABLES}/districts/{state.lower().replace(' ', '_')}_{month}_top_districts.csv"
        district_summary.to_csv(district_file, index=False)
        print("Saved:", district_file)

        # PIN codes
        pin_summary = (
            subset.groupby("pincode")["total_updates"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        top_pins = pin_summary.head(15).copy()
        total = pin_summary["total_updates"].sum()
        top_pins["share_pct"] = (top_pins["total_updates"] / total) * 100

        pin_file = f"{OUTPUT_TABLES}/pincodes/{state.lower().replace(' ', '_')}_{month}_top_pincodes.csv"
        top_pins.to_csv(pin_file, index=False)
        print("Saved:", pin_file)

    return top_rows


# =============================
# ENROLMENT INTEGRATION
# =============================

def integrate_enrolment(state_month, enrol_df):
    print("\n--- Enrolment Integration ---")

    enrol_state_month = (
        enrol_df.groupby(["state_clean", "year_month"])["total_enrollments"]
        .sum()
        .reset_index()
    )

    comparison = pd.merge(
        state_month,
        enrol_state_month,
        on=["state_clean", "year_month"],
        how="left"
    )

    comparison["update_to_enrol_ratio"] = (
        comparison["total_updates"] /
        comparison["total_enrollments"].replace(0, 1)
    )

    comparison.to_csv(
        f"{OUTPUT_TABLES}/demographic_vs_enrolment.csv", index=False
    )

    return comparison


# =============================
# CORRELATION
# =============================

def correlation_analysis(comparison):
    print("\n--- Correlation Analysis ---")

    results = []

    for state, group in comparison.groupby("state_clean"):
        valid = group.dropna(subset=["total_updates", "total_enrollments"])
        if len(valid) < 3:
            continue

        corr = valid["total_updates"].corr(valid["total_enrollments"])
        results.append({
            "state": state,
            "months_used": len(valid),
            "correlation": round(corr, 3)
        })

    corr_df = pd.DataFrame(results).sort_values("correlation", ascending=False)
    corr_df.to_csv(
        f"{OUTPUT_TABLES}/state_update_enrolment_correlation.csv",
        index=False
    )

    print("\nTop correlations:")
    print(corr_df.head(10))

    return corr_df


# =============================
# ADVANCED METRICS
# =============================

def compute_urban_concentration(df, anomalies):
    results = []

    for _, row in anomalies.iterrows():
        state = row["state_clean"]
        month = row["year_month"]

        subset = df[
            (df["state_clean"] == state) &
            (df["year_month"] == month)
        ]

        district_sum = (
            subset.groupby("district")["total_updates"]
            .sum()
            .sort_values(ascending=False)
        )

        total = district_sum.sum()
        top3_share = (district_sum.head(3).sum() / total) * 100 if total else 0

        results.append({
            "state": state,
            "month": month,
            "top3_district_share_pct": round(top3_share, 2)
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(f"{OUTPUT_TABLES}/urban_concentration.csv", index=False)
    return df_out


def compute_spike_strength(state_month, anomalies):
    results = []

    for _, row in anomalies.iterrows():
        state = row["state_clean"]
        month = row["year_month"]

        temp = state_month[state_month["state_clean"] == state].copy()
        temp = temp.sort_values("year_month").reset_index(drop=True)

        idx = temp[temp["year_month"] == month].index
        if len(idx) == 0:
            continue

        idx = idx[0]
        baseline = temp.iloc[max(0, idx-3):idx]["total_updates"].mean()
        current = temp.loc[idx, "total_updates"]

        strength = current / baseline if baseline > 0 else None

        results.append({
            "state": state,
            "month": month,
            "baseline_avg": round(baseline, 0),
            "current_updates": int(current),
            "spike_strength": round(strength, 2) if strength else None
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(f"{OUTPUT_TABLES}/spike_strength.csv", index=False)
    return df_out


# =============================
# HEATMAP EXPORT
# =============================

def export_heatmaps(df, state_month, anomalies):
    print("\n--- Heatmap Export ---")

    # STATE HEATMAP
    state_heatmap = state_month[[
        "state_clean", "year_month",
        "total_updates", "growth_pct", "zscore"
    ]].copy()

    state_heatmap.rename(columns={
        "state_clean": "state",
        "year_month": "month",
        "total_updates": "intensity"
    }, inplace=True)

    state_heatmap["spike_score"] = (
        state_heatmap["growth_pct"].abs() *
        state_heatmap["zscore"].abs()
    ).round(2)

    state_file = f"{OUTPUT_TABLES}/heatmaps/heatmap_state.csv"
    state_heatmap.to_csv(state_file, index=False)
    print("Saved:", state_file)

    

    # DISTRICT HEATMAP (only anomaly states)
    district_rows = []

    for _, row in anomalies.iterrows():
        state = row["state_clean"]
        month = row["year_month"]

        subset = df[
            (df["state_clean"] == state) &
            (df["year_month"] == month)
        ]

        district_summary = (
            subset.groupby("district")["total_updates"]
            .sum()
            .reset_index()
        )

        district_summary["state"] = state
        district_summary["month"] = month
        district_summary.rename(columns={
            "total_updates": "intensity"
        }, inplace=True)

        district_rows.append(district_summary)

    if district_rows:
        district_heatmap = pd.concat(district_rows, ignore_index=True)
        district_file = f"{OUTPUT_TABLES}/heatmaps/heatmap_district.csv"
        district_heatmap.to_csv(district_file, index=False)
        print("Saved:", district_file)

def export_heatmap_images():
    print("\n--- Rendering Heatmap Images ---")

    heatmap_dir = f"{OUTPUT_TABLES}/heatmaps"
    image_dir = f"{OUTPUT_CHARTS}/heatmaps"
    os.makedirs(image_dir, exist_ok=True)

    # ============================
    # STATE HEATMAP
    # ============================
    state_raw = pd.read_csv(f"{heatmap_dir}/heatmap_state.csv")

    state_pivot = state_raw.pivot_table(
        index="state",
        columns="month",
        values="intensity",
        aggfunc="sum",
        fill_value=0
    )

    plt.figure(figsize=(14, 8))
    plt.imshow(np.log1p(state_pivot.values), aspect="auto" ,cmap="inferno")
    plt.colorbar(label="Total Updates")
    plt.title("State × Month Heatmap")
    plt.xlabel("Month")
    plt.ylabel("State")
    plt.xticks(range(len(state_pivot.columns)), state_pivot.columns, rotation=90)
    plt.yticks(range(len(state_pivot.index)), state_pivot.index)
    plt.tight_layout()

    state_img = f"{image_dir}/state_heatmap.png"
    plt.savefig(state_img)
    plt.close()
    print("Saved:", state_img)

    # ============================
    # DISTRICT HEATMAP
    # ============================
    district_raw = pd.read_csv(f"{heatmap_dir}/heatmap_district.csv")

    district_pivot = district_raw.pivot_table(
        index="district",
        columns="month",
        values="intensity",
        aggfunc="sum",
        fill_value=0
    )

    plt.figure(figsize=(14, 10))
    plt.imshow(np.log1p(district_pivot.values), aspect="auto",cmap="inferno")
    plt.colorbar(label="Total Updates")
    plt.title("District × Month Heatmap")
    plt.xlabel("Month")
    plt.ylabel("District")
    plt.xticks(range(len(district_pivot.columns)), district_pivot.columns, rotation=90)
    plt.yticks(range(len(district_pivot.index)), district_pivot.index)
    plt.tight_layout()

    district_img = f"{image_dir}/district_heatmap.png"
    plt.savefig(district_img)
    plt.close()
    print("Saved:", district_img)

# =============================
# MAIN PIPELINE
# =============================

def main():
    df = load_demographic_data()
    enrol_df = load_enrolment_data()

    state_month = aggregate_state_month(df)
    anomalies = detect_anomalies(state_month)

    top_anomaly_rows = hotspot_analysis(df, anomalies)

    comparison = integrate_enrolment(state_month, enrol_df)
    correlation_analysis(comparison)

    print("\n--- Urban Concentration ---")
    urban_df = compute_urban_concentration(df, top_anomaly_rows)
    print(urban_df)

    print("\n--- Spike Strength ---")
    spike_df = compute_spike_strength(state_month, top_anomaly_rows)
    print(spike_df)

    export_heatmaps(df, state_month, anomalies)
    export_heatmap_images()  


if __name__ == "__main__":
    main()
