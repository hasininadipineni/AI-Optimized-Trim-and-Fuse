import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# STEP 1: LOAD DATASET
print("Loading MCU dataset...")

df = pd.read_csv("data/mcu_trim_fuse.csv")

# STEP 2: SELECT MCU-A
df = df[df["derivative"] == "MCU-A"].copy()

print("MCU-A dataset:", df.shape)

# STEP 3: SELECT FEATURES
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
    "temp_grade",
    "bg_untrim_V"
]

target = "bg_trim_code_opt"

X = df[features]
y = df[target]
groups = df["wafer_id"]

# STEP 4: PREPROCESSING
categorical_features = ["package", "temp_grade"]

numeric_features = [
    col for col in features
    if col not in categorical_features
]

preprocessor = ColumnTransformer([
    (
        "cat",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    ),
    (
        "num",
        "passthrough",
        numeric_features
    )
])

# STEP 5: SPLIT DATA BY WAFER
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

train_wafers = set(groups.iloc[train_idx])
test_wafers = set(groups.iloc[test_idx])

print("\nTraining wafers:", len(train_wafers))
print("Testing wafers:", len(test_wafers))

print("Common wafers:", train_wafers.intersection(test_wafers))
print("No wafer overlap:", train_wafers.isdisjoint(test_wafers))

# STEP 6: CREATE MODEL
model = Pipeline([
    ("preprocessing", preprocessor),
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

# STEP 7: TRAIN MODEL
print("\nTraining AI model...")

model.fit(X_train, y_train)

print("Training completed!")

# STEP 8: PREDICTIONS
predictions = model.predict(X_test)
rounded_predictions = np.rint(predictions).astype(int)

# STEP 9: PERFORMANCE EVALUATION
mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

exact_accuracy = np.mean(
    rounded_predictions == y_test.to_numpy()
) * 100

within_one = np.mean(
    np.abs(rounded_predictions - y_test.to_numpy()) <= 1
) * 100

print("\n======================================")
print("       MCU BANDGAP MODEL RESULTS")
print("======================================")

print("MAE:", round(mae, 4), "codes")
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))
print("Exact accuracy:", round(exact_accuracy, 2), "%")
print("Within +/- 1 code:", round(within_one, 2), "%")

print("\nActual vs Predicted:")

for actual, predicted in zip(
    y_test.head(10),
    predictions[:10]
):
    print(
        f"Actual: {actual}, Predicted: {predicted:.2f}"
    )

# STEP 10: SAVE MODEL
joblib.dump(
    model,
    "mcu_bandgap_model.joblib"
)

# STEP 11: SAVE TEST PREDICTIONS
results = pd.DataFrame({
    "die_id": df.iloc[test_idx]["die_id"].values,
    "wafer_id": groups.iloc[test_idx].values,
    "Actual_Trim_Code": y_test.values,
    "Predicted_Trim_Code": rounded_predictions
})

results.to_csv(
    "mcu_bandgap_predictions.csv",
    index=False
)

print("\nTrained model saved successfully!")
print("Prediction results saved successfully!")

print("\n======================================")
print("       PROJECT MODULE COMPLETED")
print("======================================")