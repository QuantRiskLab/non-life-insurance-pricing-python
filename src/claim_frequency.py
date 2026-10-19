
import numpy as np

# Simulate annual claim counts for 1,000 insurance policies
rng = np.random.default_rng(42)

number_of_policies = 1000
annual_claim_frequency = 0.15

claim_counts = rng.poisson(
    lam=annual_claim_frequency,
    size=number_of_policies
)

total_claims = np.sum(claim_counts)
average_claim_frequency = np.mean(claim_counts)

print(f"Number of policies: {number_of_policies}")
print(f"Total claims: {total_claims}")
print(f"Average claim frequency: {average_claim_frequency:.4f}")

