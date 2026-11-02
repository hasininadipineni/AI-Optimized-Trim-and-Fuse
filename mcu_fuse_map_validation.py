# ============================================================
# AI-OPTIMIZED TRIM & FUSE
# MODULE 5: MCU FUSE MAP VALIDATION
# ============================================================

import os
import pandas as pd

# STEP 1: Check required files
required_files = [
    "mcu_generated_fuse_map.csv",
    "data/mcu_trim_fuse.csv",
    "data/mcu_fuse_schema.csv"
]

for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(
            f"Required file not found: {file}. "
            "Run Module 4 first and check the data folder."
        )

# STEP 2: Load generated fuse map
print("\nLoading generated fuse map...")
fuse = pd.read_csv("mcu_generated_fuse_map.csv")

# STEP 3: Load original dataset and schema
df = pd.read_csv("data/mcu_trim_fuse.csv")
schema = pd.read_csv("data/mcu_fuse_schema.csv")

print("Fuse map records:", len(fuse))
print("Schema fields:", len(schema))

# STEP 4: Check schema columns
required_schema_columns = ["field", "lsb", "nbits"]
missing_schema_columns = [
    col for col in required_schema_columns
    if col not in schema.columns
]

if missing_schema_columns:
    raise ValueError(
        f"Schema is missing columns: {missing_schema_columns}"
    )

# STEP 5: Join source information
source = df[[
    "die_id",
    "derivative",
    "ts_cal1_30C_counts"
]].drop_duplicates(subset=["die_id"])

fuse = fuse.merge(
    source,
    on="die_id",
    how="left",
    validate="one_to_one"
)

# STEP 6: Define trim fields
trim_columns = {
    "BG_TRIM": "bg_predicted_code",
    "HSI_TRIM": "hsi_predicted_code",
    "LDO_TRIM": "ldo_predicted_code"
}

# STEP 7: Prepare schema lookup
schema_fields = set(schema["field"].astype(str))

for field in trim_columns:
    if field not in schema_fields:
        raise ValueError(
            f"Required field '{field}' is missing from schema."
        )

# STEP 8: Parse values safely
def parse_value(value):
    value = str(value).strip()
    return int(value, 0)

# STEP 9: Validate and generate fuse maps
fuse_words = []
validation_status = []
error_messages = []

for _, row in fuse.iterrows():
    errors = []
    fuse_word = 0

    derivative = row["derivative"]

    if pd.isna(derivative):
        errors.append("Missing derivative")

    if pd.isna(row["ts_cal1_30C_counts"]):
        errors.append("Missing temperature calibration")

    # Validate trim codes
    for field, column in trim_columns.items():
        value = row[column]

        schema_row = schema[
            schema["field"] == field
        ].iloc[0]

        nbits = int(schema_row["nbits"])
        maximum = (1 << nbits) - 1

        if pd.isna(value) or float(value) != int(value):
            errors.append(field + " is not an integer")
            continue

        value = int(value)

        if value < 0 or value > maximum:
            errors.append(field + " is out of range")

    # Generate fuse word when initial checks pass
    if not errors:
        for _, field_row in schema.iterrows():
            field = str(field_row["field"])
            lsb = int(field_row["lsb"])
            nbits = int(field_row["nbits"])

            if field == "BG_TRIM":
                value = int(row["bg_predicted_code"])

            elif field == "HSI_TRIM":
                value = int(row["hsi_predicted_code"])

            elif field == "LDO_TRIM":
                value = int(row["ldo_predicted_code"])

            elif field == "TS_CAL1":
                value = int(row["ts_cal1_30C_counts"])

            elif field == "RESERVED":
                value = 0

            else:
                if derivative not in schema.columns:
                    errors.append(
                        f"Schema has no column for derivative {derivative}"
                    )
                    continue

                try:
                    value = parse_value(field_row[derivative])
                except (ValueError, TypeError):
                    errors.append(
                        f"Cannot parse value for field {field}"
                    )
                    continue

            maximum = (1 << nbits) - 1

            if value < 0 or value > maximum:
                errors.append(
                    field + " exceeds bit capacity"
                )
                continue

            fuse_word |= value << lsb

    # Save result
    if errors:
        validation_status.append("FAIL")
        error_messages.append("; ".join(errors))
        fuse_words.append("NOT GENERATED")
    else:
        validation_status.append("PASS")
        error_messages.append("None")
        fuse_words.append(
            "0x" + format(fuse_word, "012X")
        )

# STEP 10: Add validation results
fuse["fuse_map_hex_generated"] = fuse_words
fuse["validation_status"] = validation_status
fuse["validation_errors"] = error_messages

# STEP 11: Save validation report
output_file = "mcu_fuse_validation_report.csv"

fuse.to_csv(output_file, index=False)

# STEP 12: Display results
total = len(fuse)
passed = validation_status.count("PASS")
failed = validation_status.count("FAIL")

print("\n===== FUSE MAP VALIDATION RESULTS =====")
print("Total fuse maps:", total)
print("Valid fuse maps:", passed)
print("Invalid fuse maps:", failed)

if total > 0:
    print(
        "Validation success rate:",
        round(passed / total * 100, 2),
        "%"
    )

print("\nSample validation results:")
print(
    fuse[[
        "die_id",
        "fuse_map_hex_generated",
        "validation_status",
        "validation_errors"
    ]].head(10).to_string(index=False)
)

print(f"\nValidation report saved: {output_file}")
print("\nMODULE 5 COMPLETED!")