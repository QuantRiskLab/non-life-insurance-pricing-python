
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Reproducible random number generator
rng = np.random.default_rng(42)

# Generate synthetic insurance portfolio
number_of_policies = 5000

data = pd.DataFrame({
    "age": rng.integers(18, 80, number_of_policies),
    "vehicle_age": rng.integers(0, 20, number_of_policies),
    "annual_mileage": rng.integers(
        5000, 30000, number_of_policies
    ),
    "exposure": rng.uniform(
        0.5, 1.0, number_of_policies
    )
})

# Define underlying claim frequency model
linear_predictor = (
    -2.5
    + 0.015 * data["age"]
    + 0.035 * data["vehicle_age"]
    + 0.00002 * data["annual_mileage"]
)

annual_claim_rate = np.exp(linear_predictor)

# Simulate claim counts
data["claim_count"] = rng.poisson(
    annual_claim_rate * data["exposure"]
)

# Define explanatory variables
features = [
    "age",
    "vehicle_age",
    "annual_mileage"
]

X = sm.add_constant(data[features])
y = data["claim_count"]

# Fit Poisson GLM with log link and exposure offset
model = sm.GLM(
    y,
    X,
    family=sm.families.Poisson(),
    offset=np.log(data["exposure"])
)

results = model.fit()

# Predict annual claim frequency
data["predicted_frequency"] = results.predict(
    X,
    offset=np.zeros(number_of_policies)
)

# Compare observed and predicted claim frequency
observed_frequency = (
    data["claim_count"].sum() / data["exposure"].sum()
)

predicted_frequency = np.average(
    data["predicted_frequency"],
    weights=data["exposure"]
)

# Display results
print(results.summary())

print(
    f"Observed annual claim frequency: "
    f"{observed_frequency:.4f}"
)

print(
    f"Predicted annual claim frequency: "
    f"{predicted_frequency:.4f}"
)

# Save synthetic dataset and predictions
output_file = (
    __import__("pathlib").Path(__file__)
    .resolve()
    .parent.parent
    / "frequency_glm_results.csv"
)

data.to_csv(output_file, index=False)

print(f"Results saved to: {output_file}")
