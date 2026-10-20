
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


# Evaluate baseline predictions
baseline_mae = mean_absolute_error(
    test["claim_amount"],
    baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        test["claim_amount"],
        baseline_predictions
    )
)

# Compare Gamma GLM against baseline
print("\nGamma Severity Model Comparison")
print("--------------------------------------------")
print(f"{'Metric':<12} {'Gamma GLM':>15} {'Baseline':>15}")
print(f"{'MAE':<12} {mae:>15.2f} {baseline_mae:>15.2f}")
print(f"{'RMSE':<12} {rmse:>15.2f} {baseline_rmse:>15.2f}")

# Calculate percentage improvement in MAE
mae_improvement = (
    (baseline_mae - mae) / baseline_mae * 100
    if baseline_mae > 0 else 0.0
)

print(f"\nMAE improvement over baseline: {mae_improvement:.2f}%")

if mae < baseline_mae:
    print("The Gamma GLM outperforms the baseline on MAE.")
else:
    print("The baseline matches or outperforms the Gamma GLM on MAE.")
