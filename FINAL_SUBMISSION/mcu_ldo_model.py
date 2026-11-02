# ============================================================
# AI-OPTIMIZED TRIM & FUSE
# MODULE 3: MCU LDO TRIM PREDICTION
# ============================================================

import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# STEP 1: Load dataset
print("\nLoading MCU dataset...")
df = pd.read_csv("data/mcu_trim_fuse.csv")

# STEP 2: Select MCU-A
df = df[df["derivative"] == "MCU-A"].copy()
print("MCU-A dataset:", df.shape)

# STEP 3: Select input features
features = [
    "die_x", "die_y", "x_mm", "y_mm", "radius_mm",
    "pcm_vth_n_mV", "pcm_vth_p_mV", "pcm_rs_ohm_sq",
    "pcm_tox_A", "pcm_mobility_rel",
    "package", "temp_grade", "ldo_untrim_V"
]

target = "ldo_trim_code_opt"

X = df[features]
y = df[target]
groups = df["wafer_id"]

# STEP 4: Preprocessing
numeric_features = [
    col for col in features
    if col not in ["package", "temp_grade"]
]
categorical_features = ["package", "temp_grade"]

preprocessor = ColumnTransformer([
    ("num", "passthrough", numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"),
     categorical_features)
])

# STEP 5: Split by wafer
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(splitter.split(X, y, groups))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("Training wafers:", groups.iloc[train_idx].nunique())
print("Testing wafers:", groups.iloc[test_idx].nunique())

# STEP 6: Create model
model = Pipeline([
    ("preprocessing", preprocessor),
    ("regressor", GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ))
])

# STEP 7: Train
print("\nTraining LDO AI model...")
model.fit(X_train, y_train)

# STEP 8: Predict
predictions = model.predict(X_test)

# STEP 9: Evaluate
mae = mean_absolute_error(y_test, predictions)
rounded_predictions = predictions.round().astype(int)

exact = (rounded_predictions == y_test).mean() * 100
within_one = (
    (rounded_predictions - y_test).abs() <= 1
).mean() * 100

print("\n===== MCU LDO MODEL RESULTS =====")
print(f"MAE: {mae:.3f} codes")
print(f"Exact accuracy: {exact:.2f}%")
print(f"Within +/- 1 code: {within_one:.2f}%")

# STEP 10: Actual vs Predicted
print("\nActual vs Predicted:")
for actual, predicted in zip(y_test.iloc[:10], predictions[:10]):
    print(f"Actual: {actual}, Predicted: {predicted:.2f}")

# STEP 11: Save model
joblib.dump(model, "mcu_ldo_model.joblib")

# STEP 12: Save predictions
results = pd.DataFrame({
    "Actual_Trim_Code": y_test.values,
    "Predicted_Trim_Code": rounded_predictions
})
results.to_csv("mcu_ldo_predictions.csv", index=False)

print("\nLDO model saved successfully!")
print("LDO predictions saved successfully!")
print("\nMODULE 3 COMPLETED!")