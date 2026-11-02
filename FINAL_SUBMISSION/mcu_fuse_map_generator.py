# ============================================================
# AI-OPTIMIZED TRIM & FUSE
# MODULE 4: MCU FUSE MAP GENERATOR
# ============================================================

import pandas as pd
import joblib

# STEP 1: Load dataset
print("\nLoading MCU dataset...")
df = pd.read_csv("data/mcu_trim_fuse.csv")

# STEP 2: Select MCU-A records
df = df[df["derivative"] == "MCU-A"].copy()
print("MCU-A dataset:", df.shape)

# STEP 3: Load trained AI models
print("\nLoading trained AI models...")

bg_model = joblib.load("../mcu_bandgap_model.joblib")
hsi_model = joblib.load("mcu_hsi_model.joblib")
ldo_model = joblib.load("mcu_ldo_model.joblib")

print("All trained models loaded successfully!")

# STEP 4: Define common input features
common_features = [
    "die_x",
    "die_y",
    "x_mm",
    "y_mm",
    "radius_mm",
    "pcm_vth_n_mV",
    "pcm_vth_p_mV",
    "pcm_rs_ohm_sq",
    "pcm_tox_A",
    "pcm_mobility_rel",
    "package",
    "temp_grade"
]

# STEP 5: Prepare model-specific features
bg_features = common_features + ["bg_untrim_V"]
hsi_features = common_features + ["hsi_untrim_MHz"]
ldo_features = common_features + ["ldo_untrim_V"]

# STEP 6: Predict trim codes
print("\nPredicting trim codes...")

df["bg_predicted_code"] = (
    bg_model.predict(df[bg_features]).round().astype(int)
)

df["hsi_predicted_code"] = (
    hsi_model.predict(df[hsi_features]).round().astype(int)
)

df["ldo_predicted_code"] = (
    ldo_model.predict(df[ldo_features]).round().astype(int)
)

# STEP 7: Display predictions
print("\n===== MCU TRIM PREDICTIONS =====")

print(
    df[[
        "die_id",
        "bg_predicted_code",
        "hsi_predicted_code",
        "ldo_predicted_code"
    ]].head(10).to_string(index=False)
)

# STEP 8: Generate fuse map
fuse_map = df[[
    "die_id",
    "bg_predicted_code",
    "hsi_predicted_code",
    "ldo_predicted_code"
]].copy()

# STEP 9: Save fuse map
output_file = "mcu_generated_fuse_map.csv"
fuse_map.to_csv(output_file, index=False)

print(f"\nFuse map saved successfully: {output_file}")
print("Total dies processed:", len(fuse_map))

# STEP 10: Compare predictions with optimal trim codes
print("\n===== PREDICTION COMPARISON =====")

comparisons = [
    ("Bandgap", "bg_predicted_code", "bg_trim_code_opt"),
    ("HSI", "hsi_predicted_code", "hsi_trim_code_opt"),
    ("LDO", "ldo_predicted_code", "ldo_trim_code_opt")
]

for name, predicted_col, target_col in comparisons:
    accuracy = (
        (df[predicted_col] - df[target_col]).abs() <= 1
    ).mean() * 100

    print(f"{name} within +/- 1 code: {accuracy:.2f}%")

print("\nMODULE 4 COMPLETED!")