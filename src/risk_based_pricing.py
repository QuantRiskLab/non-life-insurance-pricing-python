
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Reproducible random number generator
rng = np.random.default_rng(42)

# Portfolio settings
number_of_policies = 5000
expense_loading = 0.20
risk_loading = 0.10

features = [
    "age",
    "vehicle_age",
    "annual_mileage"
]

# Generate synthetic policyholder data
policies = pd.DataFrame({
    "age": rng.integers(18, 80, number_of_policies),
    "vehicle_age": rng.integers(0, 20, number_of_policies),
    "annual_mileage": rng.integers(
        5000, 30000, number_of_policies
    ),
    "exposure": rng.uniform(
        0.5, 1.0, number_of_policies
    )
})

# Generate claim frequency data
frequency_linear_predictor = (
    -2.5
    + 0.015 * policies["age"]
    + 0.035 * policies["vehicle_age"]
    + 0.00002 * policies["annual_mileage"]
)

annual_claim_rate = np.exp(frequency_linear_predictor)

policies["claim_count"] = rng.poisson(
    annual_claim_rate * policies["exposure"]
)

# Generate claim-level severity data
claim_policy_indices = np.repeat(
    np.arange(number_of_policies),
    policies["claim_count"].to_numpy()
)

claims = policies.iloc[
    claim_policy_indices
][features].reset_index(drop=True)

severity_linear_predictor = (
    7.5
    + 0.005 * claims["age"]
    + 0.025 * claims["vehicle_age"]
    + 0.00001 * claims["annual_mileage"]
)

expected_claim_amount = np.exp(
    severity_linear_predictor
)

gamma_shape = 2.0

claims["claim_amount"] = rng.gamma(
    shape=gamma_shape,
    scale=expected_claim_amount / gamma_shape
)

# Fit Poisson frequency GLM
X_frequency = sm.add_constant(
    policies[features],
    has_constant="add"
)

frequency_model = sm.GLM(
    policies["claim_count"],
    X_frequency,
    family=sm.families.Poisson(),
    offset=np.log(policies["exposure"])
).fit()

# Fit Gamma severity GLM
if claims.empty:
    raise ValueError(
        "No claims generated for severity modeling."
    )

X_severity = sm.add_constant(
    claims[features],
    has_constant="add"
)

severity_model = sm.GLM(
    claims["claim_amount"],
    X_severity,
    family=sm.families.Gamma(
        link=sm.families.links.Log()
    )
).fit()

# Predict annual claim frequency per policy
policies["predicted_frequency"] = (
    frequency_model.predict(
        X_frequency,
        offset=np.zeros(number_of_policies)
    )
)

# Predict expected claim severity per policy
policies["predicted_severity"] = (
    severity_model.predict(X_frequency)
)

# Calculate risk-based annual premiums
policies["pure_premium"] = (
    policies["predicted_frequency"]
    * policies["predicted_severity"]
)

policies["gross_premium"] = (
    policies["pure_premium"]
    * (1 + risk_loading)
    / (1 - expense_loading)
)

# Calculate portfolio-level results
average_pure_premium = policies[
    "pure_premium"
].mean()

average_gross_premium = policies[
    "gross_premium"
].mean()

expected_annual_losses = (
    policies["pure_premium"]
    * policies["exposure"]
).sum()

# Display results
print("RISK-BASED INSURANCE PRICING")
print("-" * 40)

print(
    f"Number of policies: {number_of_policies}"
)

print(
    f"Average pure annual premium: "
    f"EUR {average_pure_premium:,.2f}"
)

print(
    f"Average gross annual premium: "
    f"EUR {average_gross_premium:,.2f}"
)

print(
    f"Expected portfolio losses: "
    f"EUR {expected_annual_losses:,.2f}"
)

print("\nSample policy-level premiums:")

print(
    policies[
        features
        + [
            "predicted_frequency",
            "predicted_severity",
            "pure_premium",
            "gross_premium"
        ]
    ].head(10).round(2).to_string(index=False)
)

# Save results
output_file = (
    Path(__file__).resolve().parent.parent
    / "risk_based_pricing_results.csv"
)

policies.to_csv(output_file, index=False)

print(f"\nResults saved to: {output_file}")
