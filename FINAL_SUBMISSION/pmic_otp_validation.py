import pandas as pd

# 1. Load PMIC OTP dataset
df = pd.read_csv("data/pmic_otp_config_requests.csv")

print("PMIC OTP dataset loaded successfully!")
print("Total requests:", len(df))

# 2. Define voltage limits
voltage_limits = {
    "buck1_V": (0.6, 1.4),
    "buck2_V": (0.8, 1.6),
    "ldo1_V": (1.0, 3.3),
    "ldo2_V": (1.0, 3.6)
}

# 3. Define required power rails
expected_sequence = [
    "buck1", "buck2", "ldo1", "ldo2"
]

# 4. Define reserved I2C addresses
reserved_addresses = {
    0x00, 0x01, 0x02, 0x03,
    0x04, 0x05, 0x06, 0x07,
    0x78, 0x79, 0x7A, 0x7B,
    0x7C, 0x7D, 0x7E, 0x7F
}

# 5. Validate requests
results = []

for _, row in df.iterrows():

    errors = []

    # Check voltage limits
    for column, limits in voltage_limits.items():

        value = row[column]

        if not limits[0] <= value <= limits[1]:
            errors.append(column + " out of range")

    # Check power-up sequence
    sequence = str(row["sequence"]).split(">")

    if (
        len(sequence) != 4
        or sorted(sequence) != sorted(expected_sequence)
    ):
        errors.append("Invalid sequence")

    # Check I2C address
    address = int(str(row["i2c_addr"]), 16)

    if address in reserved_addresses:
        errors.append("Reserved I2C address")

    # Check I2C address collision
    other_addresses = str(
        row["other_addresses_on_bus"]
    ).split(";")

    other_addresses = {
        int(addr.strip(), 16)
        for addr in other_addresses
        if addr.strip()
    }

    if address in other_addresses:
        errors.append("I2C address collision")

    # Determine validation status
    if errors:
        status = "FAIL"
    else:
        status = "PASS"

    results.append({
        "request_id": row["request_id"],
        "predicted_status": status,
        "validation_errors": "; ".join(errors),
        "original_label": row["label"]
    })

# 6. Generate validation report
report = pd.DataFrame(results)

report.to_csv(
    "pmic_otp_validation_report.csv",
    index=False
)

# 7. Display results
print("\n========== PMIC OTP VALIDATION ==========")

print("\nValidation results:")
print(report["predicted_status"].value_counts())

print("\nComparison with original labels:")
print(pd.crosstab(
    report["original_label"],
    report["predicted_status"]
))

# 8. Calculate valid-request detection
valid_rows = report[
    report["original_label"] == "valid"
]

correct_valid = (
    valid_rows["predicted_status"] == "PASS"
).sum()

print("\nCorrectly accepted valid requests:",
      correct_valid, "/", len(valid_rows))

# 9. Completion
print("\nPMIC OTP VALIDATION COMPLETED!")
print("Report saved: pmic_otp_validation_report.csv")