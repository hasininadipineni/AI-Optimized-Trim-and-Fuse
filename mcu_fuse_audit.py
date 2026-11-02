import pandas as pd

# 1. Load audit dataset
df = pd.read_csv("data/mcu_fuse_map_audit.csv")

print("MCU Fuse Audit Dataset Loaded!")
print("Dataset shape:", df.shape)

# 2. Check hexadecimal fuse map format
def validate_hex(value):

    try:
        if not isinstance(value, str):
            return False

        if not value.startswith("0x"):
            return False

        int(value, 16)
        return True

    except ValueError:
        return False


df["hex_format_valid"] = df["fuse_map_hex"].apply(validate_hex)

# 3. Detect audit errors
df["predicted_has_error"] = (
    (~df["hex_format_valid"]) |
    (df["error_type"] != "none")
)

# 4. Compare predicted errors with original labels
df["audit_match"] = (
    df["predicted_has_error"] == df["has_error"]
)

# 5. Display audit results
print("\n========== MCU FUSE AUDIT RESULTS ==========")

print("\nOriginal error distribution:")
print(df["has_error"].value_counts())

print("\nError types:")
print(df["error_type"].value_counts())

print("\nHex format results:")
print(df["hex_format_valid"].value_counts())

accuracy = df["audit_match"].mean() * 100

print("\nAudit agreement:", round(accuracy, 2), "%")

# 6. Create audit report
df.to_csv(
    "mcu_fuse_audit_report.csv",
    index=False
)

# 7. Display detected error records
print("\nDetected error samples:")

print(
    df[df["predicted_has_error"] == True][
        ["die_id", "fuse_map_hex", "error_type"]
    ].head(10).to_string(index=False)
)

print("\nMCU FUSE AUDIT COMPLETED!")
print("Report saved: mcu_fuse_audit_report.csv")