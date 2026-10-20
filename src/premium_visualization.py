
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


# Define project paths
project_root = Path(__file__).resolve().parent.parent

input_file = project_root / "risk_based_pricing_results.csv"
output_directory = project_root / "figures"
output_directory.mkdir(parents=True, exist_ok=True)

# Load risk-based pricing results
policies = pd.read_csv(input_file)

required_columns = [
    "age",
    "vehicle_age",
    "annual_mileage",
    "gross_premium"
]

if not set(required_columns).issubset(policies.columns):
    raise ValueError("Required pricing columns are missing.")

# Create age groups
policies["age_group"] = pd.cut(
    policies["age"],
    bins=[17, 30, 40, 50, 60, 80],
    labels=["18-30", "31-40", "41-50", "51-60", "61-79"]
)

# Calculate average gross premiums
age_premiums = policies.groupby(
    "age_group", observed=True
)["gross_premium"].mean()

vehicle_premiums = policies.groupby(
    "vehicle_age"
)["gross_premium"].mean()

# Create premium visualization
fig, axes = plt.subplots(
    1, 3, figsize=(16, 5)
)

# Premium by driver age
axes[0].bar(
    age_premiums.index.astype(str),
    age_premiums.values,
    color="steelblue"
)

axes[0].set_title("Premium by Driver Age")
axes[0].set_xlabel("Age Group")
axes[0].set_ylabel("Average Annual Gross Premium (EUR)")

# Premium by vehicle age
axes[1].plot(
    vehicle_premiums.index,
    vehicle_premiums.values,
    marker="o",
    color="darkorange"
)

axes[1].set_title("Premium by Vehicle Age")
axes[1].set_xlabel("Vehicle Age (Years)")
axes[1].set_ylabel("Average Annual Gross Premium (EUR)")

# Premium by annual mileage
axes[2].scatter(
    policies["annual_mileage"],
    policies["gross_premium"],
    alpha=0.25,
    s=10,
    color="seagreen"
)

axes[2].set_title("Premium by Annual Mileage")
axes[2].set_xlabel("Annual Mileage (km)")
axes[2].set_ylabel("Annual Gross Premium (EUR)")

for ax in axes:
    ax.grid(alpha=0.2)

fig.suptitle(
    "Risk-Based Insurance Pricing | Poisson-Gamma GLM",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()

# Save visualization
output_file = (
    output_directory / "risk_based_premium_analysis.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Visualization saved to: {output_file}")
