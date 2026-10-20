
"""
Poisson GLM Frequency Model Validation

This script evaluates out-of-sample claim frequency
predictions using synthetic motor insurance data.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# Generate synthetic insurance policy data
n_policies = 5000

data = pd.DataFrame({
    "driver_age": rng.integers(18, 80, n_policies),
    "vehicle_age": rng.integers(0, 20, n_policies),
    "annual_mileage": rng.integers(5000, 30000, n_policies),
    "exposure": rng.uniform(0.5, 1.0, n_policies),
})

# Simulate claim counts
linear_predictor = (
    -2.5
    + 0.015 * data["driver_age"]
    + 0.035 * data["vehicle_age"]
    + 0.00002 * data["annual_mileage"]
)

data["claim_count"] = rng.poisson(
    np.exp(linear_predictor) * data["exposure"]
)

# Split data into training and testing sets
train, test = train_test_split(
    data,
    test_size=0.2,
    random_state=RANDOM_SEED
)

features = ["driver_age", "vehicle_age", "annual_mileage"]

X_train = sm.add_constant(train[features])
X_test = sm.add_constant(test[features])

# Fit Poisson GLM with exposure offset
model = sm.GLM(
    train["claim_count"],
    X_train,
    family=sm.families.Poisson(),
    offset=np.log(train["exposure"])
)

results = model.fit()

# Predict expected claim counts on unseen data
predicted_claims = results.predict(
    X_test,
    offset=np.log(test["exposure"])
)

# Evaluate predictive accuracy
mae = mean_absolute_error(
    test["claim_count"],
    predicted_claims
)

rmse = np.sqrt(
    mean_squared_error(
        test["claim_count"],
        predicted_claims
    )
)

actual_frequency = (
    test["claim_count"].sum() / test["exposure"].sum()
)

predicted_frequency = (
    predicted_claims.sum() / test["exposure"].sum()
)

print("Poisson GLM - Out-of-Sample Validation")
print("-----------------------------------")
print(f"Training policies: {len(train)}")
print(f"Testing policies: {len(test)}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Actual annual claim frequency: {actual_frequency:.4f}")
print(f"Predicted annual claim frequency: {predicted_frequency:.4f}")

# Baseline: constant claim frequency estimated from training data
baseline_annual_frequency = (
    train["claim_count"].sum() / train["exposure"].sum()
)

baseline_predictions = (
    baseline_annual_frequency * test["exposure"]
)

baseline_mae = mean_absolute_error(
    test["claim_count"],
    baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        test["claim_count"],
        baseline_predictions
    )
)

# Compare GLM and baseline performance
print("\nModel Comparison")
print("-----------------------------------")
print(f"{'Metric':<12} {'Poisson GLM':>15} {'Baseline':>15}")
print(f"{'MAE':<12} {mae:>15.4f} {baseline_mae:>15.4f}")
print(f"{'RMSE':<12} {rmse:>15.4f} {baseline_rmse:>15.4f}")

mae_improvement = (
    (baseline_mae - mae) / baseline_mae * 100
    if baseline_mae > 0 else 0.0
)

print(f"\nMAE improvement over baseline: {mae_improvement:.2f}%")

if mae < baseline_mae:
    print("The Poisson GLM outperforms the baseline on MAE.")
else:
    print("The baseline matches or outperforms the Poisson GLM on MAE.")
