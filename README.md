
# Non-Life Insurance Pricing & Risk Analytics

An actuarial Python project for insurance claim simulation, premium calculation, Generalized Linear Models (GLMs), and quantitative portfolio risk analysis.

The project combines traditional actuarial pricing methods with statistical modeling and Monte Carlo simulation to demonstrate practical applications of non-life insurance mathematics.

## Project Overview

This project implements an end-to-end actuarial modeling workflow using synthetic insurance data.

The main components include:

- Claim frequency simulation using the Poisson distribution
- Claim severity simulation using the Gamma distribution
- Pure and gross premium calculation
- Monte Carlo simulation of aggregate annual portfolio losses
- Value at Risk (VaR) and Expected Shortfall (ES)
- Poisson GLM for claim frequency modeling
- Gamma GLM for claim severity modeling
- Individual risk-based insurance premium calculation
- Out-of-sample model validation
- Baseline model comparisons
- Data visualization and automated testing

The project is designed as a quantitative portfolio demonstration relevant to actuarial science, insurance pricing, and risk management.

## Mathematical Framework

### 1. Claim Frequency

The annual number of claims for policyholder $i$ is modeled as:

$$
N_i \sim \operatorname{Poisson}(\lambda)
$$

where:

- $N_i$: annual claim count for policyholder $i$
- $\lambda = 0.15$: expected annual claim frequency

The illustrative portfolio contains 1,000 policies.

The expected total annual claim count is:

$$
E[N] = 1000 \times 0.15 = 150
$$

### 2. Claim Severity

Individual claim amounts follow a Gamma distribution:

$$
X_j \sim \operatorname{Gamma}(\alpha,\theta)
$$

with:

- Shape parameter: $\alpha = 2$
- Scale parameter: $\theta = 1500$

Expected claim severity:

$$
E[X] = \alpha\theta = 3000
$$

The variance of claim severity is:

$$
\operatorname{Var}(X) = \alpha\theta^2
$$

### 3. Aggregate Loss Model

Annual aggregate portfolio losses are modeled using a compound Poisson process:

$$
S = \sum_{j=1}^{N} X_j
$$

where $N$ denotes the total number of claims in the portfolio.

Assuming independent claim frequencies and severities:

$$
E[S] = E[N]E[X]
$$

The theoretical expected annual portfolio loss is:

$$
E[S] = 150 \times 3000 = 450000
$$

Therefore, the expected annual portfolio loss is **EUR 450,000**.

### 4. Insurance Premium Calculation

The pure premium represents the expected annual claim cost per policy:

$$
P_{\text{pure}} = \lambda E[X]
$$

$$
P_{\text{pure}} = 0.15 \times 3000 = 450
$$

The illustrative gross premium includes:

- A 10% risk loading on the pure premium
- A 20% expense loading expressed as a percentage of the gross premium

The gross premium is calculated as:

$$
P_{\text{gross}}
=
\frac{P_{\text{pure}}(1+0.10)}{1-0.20}
$$

$$
P_{\text{gross}} = 618.75
$$

The resulting illustrative gross premium is **EUR 618.75 per policy**.

### 5. Monte Carlo Risk Analysis

The project simulates 10,000 annual aggregate portfolio loss scenarios.

The 99% Value at Risk is defined as:

$$
\operatorname{VaR}_{0.99}(S)
=
F_S^{-1}(0.99)
$$

The 99% Expected Shortfall is estimated from simulated losses in the upper tail:

$$
\operatorname{ES}_{0.99}(S)
=
E[S \mid S \geq \operatorname{VaR}_{0.99}(S)]
$$

VaR provides a loss quantile at a specified confidence level, while Expected Shortfall measures the average severity of losses beyond the corresponding threshold.

## Monte Carlo Risk Visualization

![Monte Carlo Aggregate Loss Distribution](aggregate_loss_distribution.png)

The visualization illustrates the simulated distribution of annual aggregate portfolio losses, including:

- Simulated aggregate loss outcomes
- Expected portfolio losses
- 99% Value at Risk
- 99% Expected Shortfall

This analysis demonstrates how simulation can support quantitative insurance risk assessment.

## Generalized Linear Models

### 1. Claim Frequency GLM

A Poisson Generalized Linear Model estimates claim frequency using synthetic policyholder characteristics.

The explanatory variables include:

- Driver age
- Vehicle age
- Annual mileage
- Policy exposure

A log link function is used to model expected claim counts.

For policyholder $i$:

$$
\log(E[N_i \mid \mathbf{x}_i])
=
\log(e_i) + \beta_0 + \sum_{k=1}^{p}\beta_k x_{ik}
$$

where:

- $e_i$: policy exposure
- $x_{ik}$: explanatory variables
- $\beta_k$: regression coefficients

The logarithm of exposure is incorporated as an offset.

### 2. Claim Severity GLM

A Gamma Generalized Linear Model with a log link estimates expected claim severity.

For a claim with explanatory variables $\mathbf{x}_i$:

$$
\log(E[X_i \mid \mathbf{x}_i])
=
\gamma_0 + \sum_{k=1}^{p}\gamma_k x_{ik}
$$

where $\gamma_k$ denotes the severity model coefficients.

### 3. Risk-Based Insurance Pricing

Predicted annual claim frequency and expected claim severity are combined to estimate individual pure premiums.

$$
P_{\text{pure},i}
=
\widehat{\lambda}_i
\widehat{\mu}_i
$$

where:

- $\widehat{\lambda}_i$: predicted annual claim frequency
- $\widehat{\mu}_i$: predicted expected claim severity

Individual gross premiums are calculated using the illustrative loading structure:

$$
P_{\text{gross},i}
=
\frac{P_{\text{pure},i}(1+0.10)}{1-0.20}
$$

This approach demonstrates how statistical models can support differentiated insurance pricing.

## Risk-Based Premium Visualization

![Risk-Based Premium Analysis](figures/risk_based_premium_analysis.png)

The visualization explores how estimated gross premiums vary with:

- Driver age
- Vehicle age
- Annual mileage

The results are generated from synthetic data and illustrate the relationship between modeled risk characteristics and insurance premiums.

## Model Validation

The project includes separate validation scripts for claim frequency and claim severity.

Both validation workflows use an 80/20 training and testing split to assess predictive performance on held-out synthetic observations.

The evaluation includes:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Comparisons between observed and predicted outcomes

The validation scripts support the assessment of predictive accuracy and provide a foundation for further model diagnostics.

## Technologies

- Python
- NumPy
- Pandas
- SciPy
- Statsmodels
- Scikit-learn
- Matplotlib
- GitHub Actions
- Git and GitHub

## Project Structure

```text
non-life-insurance-pricing-python/
├── .github/
│   └── workflows/
│       └── python-tests.yml
├── figures/
│   └── risk_based_premium_analysis.png
├── src/
│   ├── claim_frequency.py
│   ├── claim_severity.py
│   ├── premium_calculation.py
│   ├── monte_carlo_risk.py
│   ├── risk_visualization.py
│   ├── frequency_glm.py
│   ├── severity_glm.py
│   ├── risk_based_pricing.py
│   ├── premium_visualization.py
│   ├── frequency_model_validation.py
│   └── severity_model_validation.py
├── tests/
│   └── test_actuarial_models.py
├── aggregate_loss_distribution.png
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/QuantRiskLab/non-life-insurance-pricing-python.git
cd non-life-insurance-pricing-python
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Simulate insurance claims

```bash
python src/claim_frequency.py
python src/claim_severity.py
```

### 2. Calculate insurance premiums

```bash
python src/premium_calculation.py
```

### 3. Perform Monte Carlo risk analysis

```bash
python src/monte_carlo_risk.py
```

### 4. Fit Generalized Linear Models

```bash
python src/frequency_glm.py
python src/severity_glm.py
```

### 5. Calculate risk-based premiums

```bash
python src/risk_based_pricing.py
```

### 6. Validate predictive models

```bash
python src/frequency_model_validation.py
python src/severity_model_validation.py
```

### 7. Generate visualizations

```bash
python src/risk_visualization.py
python src/premium_visualization.py
```

The visualization scripts save generated figures in the `figures/` directory.

### 8. Run unit tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Automated Testing and Continuous Integration

GitHub Actions provides an automated workflow for checking the Python project.

The workflow executes model scripts and unit tests and generates analytical outputs.

Successful workflow execution helps verify that the configured scripts run without errors in the continuous integration environment.

Generated outputs are made available as workflow artifacts.

## Model Assumptions and Limitations

This project is an educational actuarial modeling exercise based entirely on synthetic data.

The simplified modeling framework assumes:

- Independent claim occurrences and severities
- Constant baseline claim frequency and severity parameters
- Parametric Poisson and Gamma distributions
- Simplified premium loading assumptions
- Synthetic policyholder characteristics
- No explicit treatment of inflation, claims development, or reinsurance

The GLM results depend on the synthetic data-generating process and should not be interpreted as empirical findings about real insurance policyholders.

The project is not a production-ready insurance pricing system and does not provide commercially calibrated premiums.

## Future Extensions

Potential improvements include:

- Advanced cross-validation and model stability analysis
- GLM residual diagnostics and goodness-of-fit testing
- Sensitivity and scenario analysis
- Parameter uncertainty and bootstrap methods
- More comprehensive unit and integration tests
- Comparison of alternative frequency and severity models
- Portfolio profitability and loss ratio analysis
- Interactive actuarial analytics dashboards
- Reinsurance and capital modeling extensions

## Project Purpose

This project demonstrates the practical application of probability theory, statistical modeling, actuarial mathematics, and Python programming to non-life insurance pricing and quantitative risk analysis.

It was developed as an independent portfolio project to strengthen and demonstrate technical skills relevant to actuarial science, insurance analytics, and quantitative risk management.
