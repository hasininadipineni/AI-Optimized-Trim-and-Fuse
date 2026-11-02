import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load PMIC dataset
df = pd.read_csv("data/pmic_trim_otp.csv")

print("PMIC dataset loaded successfully!")
print("Original dataset shape:", df.shape)

# 2. Select PMIC-P1 derivative
df = df[df["derivative"] == "PMIC-P1"].copy()

# 3. Select input features
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
    "package",
    "ilim_untrim_A"
]

# Target column
target = "ilim_trim_code_opt"

# 4. Check required columns
required = features + [target, "wafer_id"]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

# 5. Remove missing values
df = df.dropna(subset=required)

print("Selected PMIC records:", len(df))

# 6. Separate input and target
X = df[features]
y = df[target]

# 7. Split data by wafer
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=df["wafer_id"])
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

# 8. Identify feature types
categorical = ["package"]

numeric = [
    col for col in features
    if col not in categorical
]

# 9. Preprocessing
preprocessor = ColumnTransformer([
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        categorical
    ),
    (
        "numeric",
        "passthrough",
        numeric
    )
])

# 10. Build AI model
model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
    )
])

# 11. Train model
model.fit(X_train, y_train)

# 12. Make predictions
predictions = model.predict(X_test)

# 13. Evaluate model
mae = mean_absolute_error(y_test, predictions)

r2 = r2_score(y_test, predictions)

rounded_predictions = predictions.round().astype(int)

exact_accuracy = (
    rounded_predictions == y_test.values
).mean() * 100

within_one = (
    abs(rounded_predictions - y_test.values) <= 1
).mean() * 100

# 14. Display results
print("\n========== PMIC ILIM MODEL RESULTS ==========")

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("MAE:", round(mae, 3))
print("R2 Score:", round(r2, 3))

print("Exact accuracy:", round(exact_accuracy, 2), "%")

print("Within ±1 code:", round(within_one, 2), "%")

# 15. Save trained model
joblib.dump(model, "pmic_ilim_model.joblib")

# 16. Save predictions
results = pd.DataFrame({
    "die_id": df.iloc[test_idx]["die_id"].values,
    "actual_trim_code": y_test.values,
    "predicted_trim_code": rounded_predictions
})

results.to_csv(
    "pmic_ilim_predictions.csv",
    index=False
)

print("\nPMIC ILIM MODEL COMPLETED!")
print("Model saved: pmic_ilim_model.joblib")
print("Predictions saved: pmic_ilim_predictions.csv")