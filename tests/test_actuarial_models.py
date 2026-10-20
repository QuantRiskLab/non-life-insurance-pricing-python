import unittest

import numpy as np


class TestActuarialModels(unittest.TestCase):

    def test_poisson_expected_claim_count(self):
        annual_frequency = 0.15
        exposure = 0.8

        expected_claims = annual_frequency * exposure

        self.assertAlmostEqual(expected_claims, 0.12)

    def test_gamma_expected_severity(self):
        shape = 2.0
        scale = 1500.0

        expected_severity = shape * scale

        self.assertAlmostEqual(expected_severity, 3000.0)

    def test_pure_premium(self):
        frequency = 0.15
        severity = 3000.0

        pure_premium = frequency * severity

        self.assertAlmostEqual(pure_premium, 450.0)

    def test_gross_premium(self):
        pure_premium = 450.0
        risk_loading = 0.10
        expense_loading = 0.20

        gross_premium = (
            pure_premium
            * (1 + risk_loading)
            / (1 - expense_loading)
        )

        self.assertAlmostEqual(gross_premium, 618.75)

    def test_expected_shortfall_at_least_var(self):
        losses = np.array([
            100, 200, 300, 400, 500,
            600, 700, 800, 900, 1000
        ])

        confidence_level = 0.90

        var = np.quantile(losses, confidence_level)
        es = losses[losses >= var].mean()

        self.assertGreaterEqual(es, var)


if __name__ == "__main__":
    unittest.main()
