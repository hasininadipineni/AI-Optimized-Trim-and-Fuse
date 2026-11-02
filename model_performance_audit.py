import os
import glob
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("=" * 65)
print("AI-OPTIMIZED TRIM & FUSE")
print("MODEL PERFORMANCE AUDIT")
print("=" * 65)

# Find prediction files in the current folder and all subfolders
base_folder = os.path.dirname(os.path.abspath(__file__))

prediction_files = glob.glob(
    os.path.join(base_folder, "**", "*_predictions.csv"),
    recursive=True
)

results = []

if not prediction_files:
    print("\nNo prediction CSV files found.")

for file in prediction_files:

    name = os.path.basename(file)

    try:
        df = pd.read_csv(file)

        actual_columns = [
            c for c in df.columns
            if c.lower().startswith("actual_")
        ]

        predicted_columns = [
            c for c in df.columns
            if c.lower().startswith("predicted_")
        ]

        if not actual_columns or not predicted_columns:
            print("\nSkipped:", name)
            continue

        actual = pd.to_numeric(
            df[actual_columns[0]], errors="coerce"
        )

        predicted = pd.to_numeric(
            df[predicted_columns[0]], errors="coerce"
        )

        valid = actual.notna() & predicted.notna()

        actual = actual[valid].to_numpy()
        predicted = predicted[valid].to_numpy()

        if len(actual) == 0:
            print("\nNo valid data:", name)
            continue

        mae = mean_absolute_error(actual, predicted)

        rmse = np.sqrt(
            mean_squared_error(actual, predicted)
        )

        r2 = (
            r2_score(actual, predicted)
            if len(actual) > 1 and len(np.unique(actual)) > 1
            else np.nan
        )

        exact_accuracy = np.nan
        within_one = np.nan

        is_discrete = any(word in name.lower() for word in [
            "trim", "bandgap", "hsi", "ldo",
            "ilim", "buck", "avs"
        ])

        if is_discrete:
            actual_rounded = np.rint(actual)
            predicted_rounded = np.rint(predicted)

            exact_accuracy = (
                np.mean(actual_rounded == predicted_rounded) * 100
            )

            within_one = (
                np.mean(
                    np.abs(actual_rounded - predicted_rounded) <= 1
                ) * 100
            )

        results.append({
            "Model": name,
            "Test_Rows": len(actual),
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4) if pd.notna(r2) else np.nan,
            "Exact_Accuracy_Percent": (
                round(exact_accuracy, 2)
                if pd.notna(exact_accuracy) else np.nan
            ),
            "Within_1_Percent": (
                round(within_one, 2)
                if pd.notna(within_one) else np.nan
            )
        })

        print("\nModel:", name)
        print("Rows:", len(actual))
        print("MAE:", round(mae, 4))
        print("RMSE:", round(rmse, 4))
        print("R2:", round(r2, 4) if pd.notna(r2) else "N/A")

        if pd.notna(exact_accuracy):
            print("Exact Accuracy:", round(exact_accuracy, 2), "%")
            print("Within +/-1:", round(within_one, 2), "%")

    except Exception as error:
        print("\nError in", name, ":", error)

# Save report in the same folder as this script
if results:

    report = pd.DataFrame(results)

    report_path = os.path.join(
        base_folder,
        "model_performance_audit_report.csv"
    )

    report.to_csv(report_path, index=False)

    print("\n" + "=" * 65)
    print("MODEL PERFORMANCE AUDIT COMPLETED")
    print("=" * 65)

    print(report.to_string(index=False))

    print("\nReport saved successfully:")
    print(report_path)

else:
    print("\nNo compatible prediction files were evaluated.")
