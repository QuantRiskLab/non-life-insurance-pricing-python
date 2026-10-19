
import numpy as np
import pandas as pd
import statsmodels.api as sm

from pathlib import Path

# Reproducible random number generator
rng = np.random.default_rng(42)

# Generate synthetic insurance claims
number_of_claims = 5000

data = pd.DataFrame({
    "age": rng.integers(18, 80, number_of_claims),
    "vehicle_age": rng.integers(0, 20, number_of_claims),
    "annual_mileage": rng.integers(
        5000, 30000, number_of_claims
    )
})

# Define underlying claim severity model
linear_predictor = (
    7.5
    + 0.005 * data["age"]
    + 0.025 * data["vehicle_age"]
    + 0.00001 * data["annual_mileage"]
)

expected_severity = np.exp(linear_predictor)

# Gamma distribution parameters
shape = 2.0
scale = expected_severity / shape

# Simulate individual claim amounts
data["claim_amount"] = rng.gamma(
    shape=shape,
    scale=scale
)

# Define explanatory variables
features = [
    "age",
    "vehicle_age",
    "annual_mileage"
]

X = sm.add_constant(data[features])
y = data["claim_amount"]

# Fit Gamma GLM with log link
model = sm.GLM(
    y,
    X,
    family=sm.families.Gamma(
        link=sm.families.links.Log()
    )
)

results = model.fit()

# Predict expected claim severity
data["predicted_severity"] = results.predict(X)

# Compare observed and predicted severity
observed_severity = data["claim_amount"].mean()
predicted_severity = data["predicted_severity"].mean()

# Display model results
print(results.summary())

print(
    f"Observed average claim severity: "
    f"EUR {observed_severity:,.2f}"
)

print(
    f"Predicted average claim severity: "
    f"EUR {predicted_severity:,.2f}"
)

# Save synthetic dataset and predictions
output_file = (
    Path(__file__).resolve().parent.parent
    / "severity_glm_results.csv"
)

data.to_csv(output_file, index=False)

print(f"Results saved to: {output_file}")
