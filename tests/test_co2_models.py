"""Verify numerical metrics, historical rates and chronological leakage checks."""
import unittest
import numpy as np
import pandas as pd
from co2_models import (
    add_rate_features,
    compare_models,
    extract_environment_measurements,
    forecast_metrics,
)


class ForecastTests(unittest.TestCase):
    """Check independent examples and the final-test split boundary contract."""

    def test_metrics_and_sign(self):
        """Check signed error and squared error against hand calculations."""
        result = forecast_metrics([400., 420.], [410., 420.])
        self.assertEqual(result['MAE_ppm'], 5.)
        self.assertEqual(result['mean_signed_error_ppm'], 5.)
        self.assertAlmostEqual(result['RMSE_ppm'], np.sqrt(50.))

    def test_invalid_metrics(self):
        """Reject absent, misaligned or nonfinite measurement arrays."""
        for actual, pred in [([], []), ([1], [1, 2]), ([np.nan], [1])]:
            with self.assertRaises(ValueError):
                forecast_metrics(actual, pred)

    def test_rate_uses_actual_elapsed_time(self):
        """Irregular 21-minute intervals must not be divided by nominal 20."""
        frame = pd.DataFrame({'issue_time':['2024-03-01T00:21:00Z'],
            'lag_time_20':['2024-03-01T00:00:00Z'], 'co2_ppm':[442.], 'co2_lag_20min':[400.]})
        self.assertEqual(add_rate_features(frame)['co2_rate_20_ppm_per_min'].iloc[0], 2.)
        frame['lag_time_20'] = frame.issue_time
        with self.assertRaises(ValueError):
            add_rate_features(frame)

    def test_environment_json_extracts_known_measurements(self):
        """Extract a known CO2 reading and ignore unrelated variables."""
        payload = '[{"variable": {"name": "Carbon dioxide"}, "values": [{"time": 10, "value": 500}]}, {"variable": {"name": "Battery"}, "values": [{"time": 10, "value": 4.2}]}]'
        self.assertEqual(
            extract_environment_measurements(payload, "sensor-1"),
            [("sensor-1", 10, "Carbon dioxide", 500)],
        )

    def test_environment_json_rejects_non_list_payload(self):
        """Reject a valid JSON object with the wrong payload structure."""
        with self.assertRaises(TypeError):
            extract_environment_measurements('{"values": []}')

    def test_target_boundary_leakage_rejected(self):
        """An overlapping target boundary must fail before any model is fitted."""
        frame = pd.DataFrame({'issue_time':['2024-03-01T00:20:00Z'],
            'lag_time_20':['2024-03-01T00:00:00Z'], 'target_time':['2024-03-01T00:40:00Z'],
            'co2_ppm':[442.], 'co2_lag_20min':[400.]})
        with self.assertRaisesRegex(ValueError, 'chronological'):
            compare_models(frame, frame, frame, 'unused_test_output')


if __name__ == '__main__':
    unittest.main()
