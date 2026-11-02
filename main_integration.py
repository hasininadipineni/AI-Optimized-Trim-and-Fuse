import os
import glob
import pandas as pd

print("=" * 55)
print(" AI-OPTIMIZED TRIM & FUSE - FINAL INTEGRATION")
print("=" * 55)

# 1. Check MCU models
mcu_models = [
    "mcu_bandgap_model.joblib",
    "mcu_hsi_model.joblib",
    "mcu_ldo_model.joblib"
]

# 2. Check PMIC models
pmic_models = [
    "pmic_ilim_model.joblib",
    "pmic_buck1_model.joblib",
    "pmic_buck2_model.joblib",
    "pmic_ldo1_model.joblib",
    "pmic_ldo2_model.joblib"
]

# 3. Check SoC models
soc_models = [
    "soc_avs_model.joblib",
    "soc_power_model.joblib"
]

# 4. Check important reports
reports = [
    "mcu_fuse_validation_report.csv",
    "mcu_fuse_audit_report.csv",
    "pmic_otp_validation_report.csv",
    "soc_memory_repair_report.csv"
]

# 5. Check model files
print("\n========== MODEL STATUS ==========")

all_models = mcu_models + pmic_models + soc_models

for model in all_models:
    if os.path.exists(model):
        print("[OK] ", model)
    else:
        print("[MISSING] ", model)

# 6. Check sensor models
print("\n========== SENSOR MODEL STATUS ==========")

sensor_models = glob.glob("sensor_*_model.joblib")

for model in sorted(sensor_models):
    print("[OK] ", model)

print("Total sensor models:", len(sensor_models))

# 7. Check reports
print("\n========== REPORT STATUS ==========")

for report in reports:
    if os.path.exists(report):
        print("[OK] ", report)
    else:
        print("[MISSING] ", report)

# 8. Count generated prediction files
prediction_files = glob.glob("*_predictions.csv")

print("\nTotal prediction files:", len(prediction_files))

# 9. Create integration summary
summary = {
    "MCU models": sum(os.path.exists(f) for f in mcu_models),
    "PMIC models": sum(os.path.exists(f) for f in pmic_models),
    "SoC models": sum(os.path.exists(f) for f in soc_models),
    "Sensor models": len(sensor_models),
    "Reports available": sum(os.path.exists(f) for f in reports),
    "Prediction files": len(prediction_files)
}

summary_df = pd.DataFrame(
    list(summary.items()),
    columns=["Module", "Completed_Count"]
)

summary_df.to_csv(
    "final_integration_report.csv",
    index=False
)

# 10. Display final summary
print("\n========== FINAL SUMMARY ==========")

print(summary_df.to_string(index=False))

print("\nFINAL INTEGRATION CHECK COMPLETED!")
print("Report saved: final_integration_report.csv")