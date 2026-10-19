
import numpy as np

# Reproducible simulation
rng = np.random.default_rng(42)

# Portfolio parameters
number_of_policies = 1000
annual_claim_frequency = 0.15
shape = 2.0
scale = 1500.0

# Monte Carlo settings
number_of_simulations = 10000
confidence_level = 0.99

# Simulate aggregate annual losses
aggregate_losses = np.zeros(number_of_simulations)

for simulation in range(number_of_simulations):
    number_of_claims = rng.poisson(
        number_of_policies * annual_claim_frequency
    )

    claim_amounts = rng.gamma(
        shape=shape,
        scale=scale,
        size=number_of_claims
    )

    aggregate_losses[simulation] = np.sum(claim_amounts)

# Risk measures
expected_loss = np.mean(aggregate_losses)

value_at_risk = np.quantile(
    aggregate_losses,
    confidence_level
)

expected_shortfall = np.mean(
    aggregate_losses[aggregate_losses >= value_at_risk]
)

# Display results
print(f"Expected annual loss: EUR {expected_loss:,.2f}")
print(f"VaR (99%): EUR {value_at_risk:,.2f}")
print(f"Expected Shortfall (99%): EUR {expected_shortfall:,.2f}")
