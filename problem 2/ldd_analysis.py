
"""
LDD Resistance Characterization and MOSFET I-V Analysis
-------------------------------------------------------
Input:
    data/measurements.csv

Output:
    results/ldd_results.csv
    results/transfer_characteristics.png
    results/output_characteristics.png
    results/ldd_resistance_fit.png

IMPORTANT:
The generated dataset is illustrative, not measured silicon data.
A valid LDD resistance extraction requires the voltage drop across
the actual LDD segment, measured or reliably estimated.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. File paths
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "measurements.csv"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. Load and validate CSV data
# ---------------------------------------------------------

REQUIRED_COLUMNS = [
    "sweep_type",
    "vg_V",
    "vds_V",
    "id_A",
    "v_ldd_start_V",
    "v_ldd_end_V",
]

NUMERIC_COLUMNS = [
    "vg_V",
    "vds_V",
    "id_A",
    "v_ldd_start_V",
    "v_ldd_end_V",
]


def load_measurements(csv_path):
    """Load measurements and validate the required columns."""

    if not csv_path.exists():
        raise FileNotFoundError(
            f"\nCSV file not found:\n{csv_path}\n\n"
            "Create a data folder beside this script and place "
            "measurements.csv inside it."
        )

    df = pd.read_csv(csv_path)

    # Remove accidental spaces around column names.
    df.columns = df.columns.str.strip()

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"CSV is missing columns: {missing_columns}\n"
            f"Columns found: {list(df.columns)}"
        )

    # Standardize sweep labels.
    df["sweep_type"] = (
        df["sweep_type"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    valid_sweeps = {"transfer", "output", "ldd"}

    unknown_sweeps = set(df["sweep_type"].dropna().unique()) - valid_sweeps

    if unknown_sweeps:
        raise ValueError(
            f"Unknown sweep types: {unknown_sweeps}. "
            "Use transfer, output, or ldd."
        )

    if df["sweep_type"].isna().any():
        raise ValueError("Some rows have a missing sweep_type.")

    # Convert measurement columns to numeric values.
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Every row needs these three terminal measurements.
    base_columns = ["vg_V", "vds_V", "id_A"]

    invalid_base = df[base_columns].isna().any(axis=1)

    if invalid_base.any():
        bad_rows = (df.index[invalid_base] + 2).tolist()
        raise ValueError(
            "Missing or invalid gate/drain measurements "
            f"in CSV row(s): {bad_rows}"
        )

    # Reject infinite values in the required terminal measurements.
    if not np.isfinite(df[base_columns].to_numpy(dtype=float)).all():
        raise ValueError(
            "Gate voltage, drain voltage, and drain current "
            "must be finite numbers."
        )

    # LDD endpoint voltages are required only for LDD rows.
    ldd_mask = df["sweep_type"].eq("ldd")
    ldd_columns = ["v_ldd_start_V", "v_ldd_end_V"]

    if df.loc[ldd_mask, ldd_columns].isna().any().any():
        bad_rows = (
            df.loc[ldd_mask]
            .index[
                df.loc[ldd_mask, ldd_columns].isna().any(axis=1)
            ] + 2
        ).tolist()

        raise ValueError(
            "LDD rows must contain both endpoint voltages. "
            f"Check CSV row(s): {bad_rows}"
        )

    if ldd_mask.any():
        ldd_values = df.loc[ldd_mask, ldd_columns].to_numpy(dtype=float)

        if not np.isfinite(ldd_values).all():
            raise ValueError(
                "LDD endpoint voltages must be finite numbers."
            )

    if (df["id_A"] < 0).any():
        raise ValueError(
            "This script expects non-negative drain current. "
            "Check the current sign convention in your CSV."
        )

    if df.empty:
        raise ValueError("The CSV contains no measurement rows.")

    print(f"Loaded {len(df)} measurement rows.")
    print(f"Input file: {csv_path}")

    return df


# ---------------------------------------------------------
# 3. Calculate LDD voltage drop and pointwise resistance
# ---------------------------------------------------------

def calculate_ldd_resistance(df):
    """Calculate resistance only for rows marked as LDD."""

    results = df.copy()

    results["delta_v_ldd_V"] = np.nan
    results["r_ldd_ohm"] = np.nan

    ldd_mask = results["sweep_type"].eq("ldd")

    # Keep the signed voltage difference.
    results.loc[ldd_mask, "delta_v_ldd_V"] = (
        results.loc[ldd_mask, "v_ldd_start_V"]
        - results.loc[ldd_mask, "v_ldd_end_V"]
    )

    # R = |V| / |I|. Resistance is undefined at zero current.
    valid_current = ldd_mask & results["id_A"].ne(0)

    results.loc[valid_current, "r_ldd_ohm"] = (
        results.loc[valid_current, "delta_v_ldd_V"].abs()
        / results.loc[valid_current, "id_A"].abs()
    )

    output_csv = RESULTS_DIR / "ldd_results.csv"
    results.to_csv(output_csv, index=False)

    print(f"Detailed results saved to: {output_csv}")

    return results


# ---------------------------------------------------------
# 4. Plot transfer characteristics: ID versus VG
# ---------------------------------------------------------

def plot_transfer_characteristics(df):
    transfer = df[df["sweep_type"] == "transfer"].copy()

    if transfer.empty:
        print("No transfer data found. Skipping transfer plot.")
        return

    plt.figure(figsize=(8, 5))

    for vds, group in transfer.groupby("vds_V"):
        group = group.sort_values("vg_V")

        plt.plot(
            group["vg_V"],
            group["id_A"] * 1e6,
            marker="o",
            markersize=4,
            label=f"VDS = {vds:g} V",
        )

    plt.xlabel("Gate-to-source voltage VG (V)")
    plt.ylabel("Drain current ID (µA)")
    plt.title("MOSFET Transfer Characteristics")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_file = RESULTS_DIR / "transfer_characteristics.png"
    plt.savefig(output_file, dpi=200)
    plt.close()

    print(f"Transfer plot saved to: {output_file}")


# ---------------------------------------------------------
# 5. Plot output characteristics: ID versus VDS
# ---------------------------------------------------------

def plot_output_characteristics(df):
    output = df[df["sweep_type"] == "output"].copy()

    if output.empty:
        print("No output data found. Skipping output plot.")
        return

    plt.figure(figsize=(8, 5))

    for vg, group in output.groupby("vg_V"):
        group = group.sort_values("vds_V")

        plt.plot(
            group["vds_V"],
            group["id_A"] * 1e6,
            marker="o",
            markersize=3,
            label=f"VG = {vg:g} V",
        )

    plt.xlabel("Drain-to-source voltage VDS (V)")
    plt.ylabel("Drain current ID (µA)")
    plt.title("MOSFET Output Characteristics")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_file = RESULTS_DIR / "output_characteristics.png"
    plt.savefig(output_file, dpi=200)
    plt.close()

    print(f"Output plot saved to: {output_file}")


# ---------------------------------------------------------
# 6. Fit LDD voltage drop against current
# ---------------------------------------------------------

def fit_ldd_resistance(df):
    """
    Fit |V_LDD,start - V_LDD,end| = R * |ID| + intercept.

    The fitted slope estimates resistance only if the measured
    voltage difference corresponds to the actual LDD segment.
    """

    ldd = df[df["sweep_type"] == "ldd"].copy()

    if ldd.empty:
        print("No LDD rows found. Cannot fit LDD resistance.")
        return None

    ldd["delta_v_abs_V"] = (
        ldd["v_ldd_start_V"] - ldd["v_ldd_end_V"]
    ).abs()

    ldd["current_abs_A"] = ldd["id_A"].abs()

    # A zero-current point cannot be used to estimate V/I.
    fit_data = ldd[
        (ldd["current_abs_A"] > 0)
        & np.isfinite(ldd["current_abs_A"])
        & np.isfinite(ldd["delta_v_abs_V"])
    ].copy()

    if len(fit_data) < 2:
        print("At least two valid nonzero-current LDD points are needed.")
        return None

    current = fit_data["current_abs_A"].to_numpy()
    voltage = fit_data["delta_v_abs_V"].to_numpy()

    # Linear model: voltage = slope * current + intercept.
    slope, intercept = np.polyfit(current, voltage, 1)

    predicted_voltage = slope * current + intercept

    ss_res = np.sum((voltage - predicted_voltage) ** 2)
    ss_tot = np.sum((voltage - np.mean(voltage)) ** 2)

    r_squared = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else float("nan")
    )

    print("\n========== LDD RESISTANCE ANALYSIS ==========")
    print(f"Valid LDD data points : {len(fit_data)}")
    print(f"Fitted resistance     : {slope:.3f} ohms")
    print(f"Fitted intercept      : {intercept:.6e} V")
    print(f"R-squared             : {r_squared:.6f}")
    print("Model: |V_LDD| = R_LDD * |ID| + intercept")
    print("=============================================\n")

    # Plot data and fitted line.
    plt.figure(figsize=(8, 5))

    plt.scatter(
        current * 1e6,
        voltage * 1e3,
        label="LDD data",
        marker="o",
        s=45,
    )

    current_line = np.linspace(current.min(), current.max(), 200)
    voltage_line = slope * current_line + intercept

    plt.plot(
        current_line * 1e6,
        voltage_line * 1e3,
        label=f"Linear fit: R = {slope:.2f} Ω",
    )

    plt.xlabel("Absolute drain current |ID| (µA)")
    plt.ylabel("Absolute LDD voltage drop (mV)")
    plt.title("LDD Resistance Extraction")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_file = RESULTS_DIR / "ldd_resistance_fit.png"
    plt.savefig(output_file, dpi=200)
    plt.close()

    print(f"LDD fit plot saved to: {output_file}")

    return {
        "resistance_ohm": slope,
        "intercept_V": intercept,
        "r_squared": r_squared,
        "number_of_points": len(fit_data),
    }


# ---------------------------------------------------------
# 7. Main program
# ---------------------------------------------------------

def main():
    print("\nMOSFET LDD Resistance Characterization")
    print("--------------------------------------")

    df = load_measurements(DATA_FILE)

    print("\nRows by sweep type:")
    print(df["sweep_type"].value_counts().to_string())

    results = calculate_ldd_resistance(df)

    plot_transfer_characteristics(df)
    plot_output_characteristics(df)

    fit = fit_ldd_resistance(df)

    print("\nAnalysis completed.")
    print(f"All results are in: {RESULTS_DIR}")

    if fit is not None:
        print(
            "\nEstimated LDD resistance: "
            f"{fit['resistance_ohm']:.3f} ohms"
        )

    print(
        "\nReminder: if the input CSV contains synthetic data, "
        "the extracted value is only a software demonstration."
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, pd.errors.ParserError) as error:
        print(f"\nERROR: {error}")
        raise SystemExit(1)
