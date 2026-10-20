
"""
Gamma GLM Severity Model Validation

Evaluate out-of-sample claim severity predictions
using synthetic motor insurance claims data.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# Generate synthetic claims data
n_claims = 5000

data = pd.DataFrame({
    "driver_age": rng.integers(18, 80, n_claims),
    "vehicle_age": rng.integers(0, 20, n_claims),
    "annual_mileage": rng.integers(5000, 30000, n_claims),
})

# Define expected claim severity
linear_predictor = (
    7.5
    + 0.005 * data["driver_age"]
    + 0.025 * data["vehicle_age"]
    + 0.00001 * data["annual_mileage"]
)

expected_severity = np.exp(linear_predictor)

# Generate Gamma-distributed claim amounts
shape_parameter = 2.0

data["claim_amount"] = rng.gamma(
    shape=shape_parameter,
    scale=expected_severity / shape_parameter
)

# Split into training and testing datasets
train, test = train_test_split(
    data,
    test_size=0.2,
    random_state=RANDOM_SEED
)

features = ["driver_age", "vehicle_age", "annual_mileage"]

X_train = sm.add_constant(train[features])
X_test = sm.add_constant(test[features])

# Fit Gamma GLM with log link
model = sm.GLM(
    train["claim_amount"],
    X_train,
    family=sm.families.Gamma(
        link=sm.families.links.Log()
    )
)

results = model.fit()

# Predict claim severity on unseen data
predicted_severity = results.predict(X_test)

# Calculate validation metrics
mae = mean_absolute_error(
    test["claim_amount"],
    predicted_severity
)

rmse = np.sqrt(
    mean_squared_error(
        test["claim_amount"],
        predicted_severity
    )
)

actual_mean = test["claim_amount"].mean()
predicted_mean = predicted_severity.mean()

print("Gamma GLM - Out-of-Sample Validation")
print("------------------------------------")
print(f"Training claims: {len(train)}")
print(f"Testing claims: {len(test)}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Actual mean claim severity: {actual_mean:.2f}")
print(f"Predicted mean claim severity: {predicted_mean:.2f}")
