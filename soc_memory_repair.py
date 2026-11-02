import pandas as pd

# 1. Load memory repair dataset
df = pd.read_csv("data/memory_repair_bitmaps.csv")

print("Memory repair dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 2. Function to calculate repair requirements
def analyze_repair(row):

    fail_cells = str(row["fail_cells"])

    failed_rows = set()
    failed_cols = set()

    if fail_cells and fail_cells != "nan":

        for cell in fail_cells.split(";"):

            if ":" in cell:
                r, c = cell.split(":")

                failed_rows.add(int(r))
                failed_cols.add(int(c))

    # Count required spare rows and columns
    required_rows = len(failed_rows)
    required_cols = len(failed_cols)

    available_rows = int(row["spare_rows"])
    available_cols = int(row["spare_cols"])

    # Check repairability using a simple
    # whole-row/whole-column repair estimate
    if (
        required_rows <= available_rows
        and required_cols <= available_cols
    ):
        status = "REPAIRABLE"
    else:
        status = "NOT REPAIRABLE"

    return pd.Series({
        "failed_row_count": required_rows,
        "failed_col_count": required_cols,
        "available_spare_rows": available_rows,
        "available_spare_cols": available_cols,
        "estimated_repair_status": status
    })

# 3. Analyze every memory array
analysis = df.apply(analyze_repair, axis=1)

# 4. Combine original and calculated results
report = pd.concat([df, analysis], axis=1)

# 5. Compare with optimal repair result
report["matches_optimal_result"] = (
    report["estimated_repair_status"] == 
    report["optimal_repairable"].map({
        True: "REPAIRABLE",
        False: "NOT REPAIRABLE"
    })
)

# 6. Display results
print("\n========== MEMORY REPAIR RESULTS ==========")

print("\nOriginal optimal repairability:")
print(df["optimal_repairable"].value_counts())

print("\nEstimated repairability:")
print(report["estimated_repair_status"].value_counts())

print("\nAgreement with optimal result:",
      round(report["matches_optimal_result"].mean() * 100, 2), "%")

# 7. Save repair report
report.to_csv(
    "soc_memory_repair_report.csv",
    index=False
)

print("\nMEMORY REPAIR ANALYSIS COMPLETED!")
print("Report saved: soc_memory_repair_report.csv")