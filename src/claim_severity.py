
import numpy as np

# Reproducible random number generator
rng = np.random.default_rng(42)

# Model parameters
number_of_claims = 1000
shape = 2.0
scale = 1500.0

# Simulate individual claim amounts using a Gamma distribution
claim_amounts = rng.gamma(
    shape=shape,
    scale=scale,
    size=number_of_claims
)

# Calculate descriptive statistics
total_claim_amount = np.sum(claim_amounts)
average_claim_amount = np.mean(claim_amounts)
maximum_claim_amount = np.max(claim_amounts)

# Display results
print(f"Number of claims: {number_of_claims}")
print(f"Total claim amount: EUR {total_claim_amount:,.2f}")
print(f"Average claim amount: EUR {average_claim_amount:,.2f}")
print(f"Maximum claim amount: EUR {maximum_claim_amount:,.2f}")
