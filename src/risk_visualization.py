
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
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

# Simulate annual aggregate losses
aggregate_losses = np.zeros(number_of_simulations)

for i in range(number_of_simulations):
    number_of_claims = rng.poisson(
        number_of_policies * annual_claim_frequency
    )

    claim_amounts = rng.gamma(
        shape=shape,
        scale=scale,
        size=number_of_claims
    )

    aggregate_losses[i] = claim_amounts.sum()

# Calculate risk measures
expected_loss = aggregate_losses.mean()
value_at_risk = np.quantile(
    aggregate_losses, confidence_level
)
expected_shortfall = aggregate_losses[
    aggregate_losses >= value_at_risk
].mean()

# Create output directory
output_dir = Path(__file__).resolve().parent.parent / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

# Plot aggregate loss distribution
fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(
    aggregate_losses,
    bins=60,
    density=True,
    alpha=0.7,
    edgecolor="white",
    label="Simulated aggregate losses"
)

ax.axvline(
    expected_loss,
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"Expected Loss: EUR {expected_loss:,.0f}"
)

ax.axvline(
    value_at_risk,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"99% VaR: EUR {value_at_risk:,.0f}"
)

ax.axvline(
    expected_shortfall,
    color="purple",
    linestyle=":",
    linewidth=2,
    label=f"99% ES: EUR {expected_shortfall:,.0f}"
)

ax.set_title("Monte Carlo Simulation of Annual Insurance Losses")
ax.set_xlabel("Aggregate Annual Losses (EUR)")
ax.set_ylabel("Probability Density")
ax.legend()
ax.grid(alpha=0.2)

fig.tight_layout()
fig.savefig(
    output_dir / "aggregate_loss_distribution.png",
    dpi=300
)
plt.close(fig)

print("Risk visualization saved successfully.")
