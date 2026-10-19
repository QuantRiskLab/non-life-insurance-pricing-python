
import numpy as np

# Reproducible random number generator
rng = np.random.default_rng(42)

# Insurance portfolio parameters
number_of_policies = 1000
annual_claim_frequency = 0.15

# Gamma claim severity parameters
shape = 2.0
scale = 1500.0

# Premium loading assumptions
expense_loading = 0.20
risk_loading = 0.10

# Simulate annual claim counts per policy
claim_counts = rng.poisson(
    lam=annual_claim_frequency,
    size=number_of_policies
)

# Simulate individual claim amounts
total_number_of_claims = int(np.sum(claim_counts))

claim_amounts = rng.gamma(
    shape=shape,
    scale=scale,
    size=total_number_of_claims
)

# Calculate simulated aggregate losses
aggregate_losses = np.sum(claim_amounts)

# Calculate theoretical expected losses
expected_claim_severity = shape * scale

expected_annual_losses = (
    number_of_policies
    * annual_claim_frequency
    * expected_claim_severity
)

# Calculate premiums per policy
pure_premium = (
    expected_annual_losses / number_of_policies
)

# Expense loading as a percentage of gross premium
# Risk loading as a percentage of pure premium
risk_adjusted_premium = pure_premium * (1 + risk_loading)

gross_premium = (
    risk_adjusted_premium / (1 - expense_loading)
)

# Display results
print(f"Number of policies: {number_of_policies}")
print(f"Simulated claims: {total_number_of_claims}")
print(f"Simulated aggregate losses: EUR {aggregate_losses:,.2f}")
print(f"Expected annual losses: EUR {expected_annual_losses:,.2f}")
print(f"Pure premium per policy: EUR {pure_premium:,.2f}")
print(f"Gross premium per policy: EUR {gross_premium:,.2f}")
