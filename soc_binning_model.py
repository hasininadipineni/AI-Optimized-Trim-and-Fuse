import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    accuracy_score
)

# 1. Load dataset
df = pd.read_csv("data/soc_binning_harvesting.csv")

print("SoC dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 2. Select SoC-Std derivative
df = df[df["derivative"] == "SoC-Std"].copy()

print("Selected records:", len(df))

# 3. Create useful features
core_fmax = [f"core{i}_fmax_GHz" for i in range(8)]
core_defects = [f"core{i}_defect" for i in range(8)]

# Count healthy cores
df["healthy_core_count"] = (
    ~df[core_defects].astype(bool)
).sum(axis=1)

# Calculate average healthy core frequency
df["average_core_fmax"] = df[core_fmax].mean(axis=1)

# 4. Input features
features = [
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
    "ro_freq_MHz",
    "cpm_path_delay_ps",
    "iddq_25C_mA",
    "healthy_core_count",
    "average_core_fmax",
    "l2_slice0_fail_bits",
    "l2_slice1_fail_bits",
    "l2_slice2_fail_bits",
    "l2_slice3_fail_bits",
    "l2_repairable",
    "derivative"
]

# 5. Targets
targets = [
    "avs_voltage_bin",
    "power_at_sku_W"
]

# 6. Check columns
required = features + targets + ["wafer_id"]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=targets + ["wafer_id"])

# 7. Split by wafer
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(df, groups=df["wafer_id"])
)

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]

X_train = train_df[features]
X_test = test_df[features]

# 8. Preprocessing
categorical = ["derivative"]

numeric = [
    c for c in features
    if c not in categorical
]

preprocessor = ColumnTransformer([
    (
        "cat",
        OneHotEncoder(handle_unknown="ignore"),
        categorical
    ),
    (
        "num",
        SimpleImputer(strategy="median"),
        numeric
    )
])

# 9. Train AVS voltage bin classifier
print("\nTraining AVS voltage bin model...")

avs_model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )
    )
])

avs_model.fit(
    X_train,
    train_df["avs_voltage_bin"]
)

avs_predictions = avs_model.predict(X_test)

avs_accuracy = accuracy_score(
    test_df["avs_voltage_bin"],
    avs_predictions
) * 100

print("AVS accuracy:", round(avs_accuracy, 2), "%")

joblib.dump(
    avs_model,
    "soc_avs_model.joblib"
)

pd.DataFrame({
    "die_id": test_df["die_id"].values,
    "actual_avs_bin": test_df["avs_voltage_bin"].values,
    "predicted_avs_bin": avs_predictions
}).to_csv(
    "soc_avs_predictions.csv",
    index=False
)

# 10. Train power prediction model
print("\nTraining SoC power model...")

power_model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
    )
])

power_model.fit(
    X_train,
    train_df["power_at_sku_W"]
)

power_predictions = power_model.predict(X_test)

power_mae = mean_absolute_error(
    test_df["power_at_sku_W"],
    power_predictions
)

power_r2 = r2_score(
    test_df["power_at_sku_W"],
    power_predictions
)

print("Power MAE:", round(power_mae, 3))
print("Power R2 Score:", round(power_r2, 3))

joblib.dump(
    power_model,
    "soc_power_model.joblib"
)

pd.DataFrame({
    "die_id": test_df["die_id"].values,
    "actual_power_W": test_df["power_at_sku_W"].values,
    "predicted_power_W": power_predictions
}).to_csv(
    "soc_power_predictions.csv",
    index=False
)

# 11. Completion
print("\nSOC BINNING AND HARVESTING COMPLETED!")
print("Saved AVS and power models successfully.")