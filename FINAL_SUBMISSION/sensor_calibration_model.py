import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load dataset
df = pd.read_csv("data/sensor_calibration.csv")

print("Sensor dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 2. Select ACC-2G derivative
df = df[df["derivative"] == "ACC-2G"].copy()

# 3. Input features
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
    "full_scale_g",
    "nominal_sens_LSB_per_g",
    "package",
    "stress_gauge_mV",
    "asic_vref_mV",
    "mems_resonance_kHz",
    "mems_q_factor"
]

# 4. Calibration targets
targets = [
    "cal_offset_code_x",
    "cal_tco_code_x",
    "cal_tcs_code_x",
    "cal_offset_code_y",
    "cal_tco_code_y",
    "cal_tcs_code_y",
    "cal_offset_code_z",
    "cal_tco_code_z",
    "cal_tcs_code_z"
]

# 5. Check required columns
required = features + targets + ["wafer_id"]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

# 6. Remove missing values
df = df.dropna(subset=required)

print("Selected records:", len(df))

# 7. Split data by wafer
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
categorical = ["package"]

numeric = [
    c for c in features
    if c not in categorical
]

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

# 9. Train separate models
for target in targets:

    print("\nTraining:", target)

    y_train = train_df[target]
    y_test = test_df[target]

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

    # Train model
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    rounded_predictions = predictions.round().astype(int)

    # Evaluate
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    exact_accuracy = (
        rounded_predictions == y_test.values
    ).mean() * 100

    within_one = (
        abs(rounded_predictions - y_test.values) <= 1
    ).mean() * 100

    print("MAE:", round(mae, 3))
    print("R2 Score:", round(r2, 3))
    print("Exact accuracy:", round(exact_accuracy, 2), "%")
    print("Within ±1 code:", round(within_one, 2), "%")

    # Save model
    model_name = target.replace("_code", "")

    joblib.dump(
        model,
        f"sensor_{model_name}_model.joblib"
    )

    # Save predictions
    results = pd.DataFrame({
        "die_id": test_df["die_id"].values,
        "actual_code": y_test.values,
        "predicted_code": rounded_predictions
    })

    results.to_csv(
        f"sensor_{model_name}_predictions.csv",
        index=False
    )

print("\nSENSOR CALIBRATION MODELS COMPLETED!")