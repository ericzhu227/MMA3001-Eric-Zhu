"""Verify numerical metrics, historical rates and chronological leakage checks."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from co2_models import (
    add_rate_features,
    build_forecast_features,
    clean_environment_readings,
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

    def test_clean_environment_readings_removes_duplicates_and_conflicts(self):
        """Remove exact repeats and exclude all conflicting sensor readings."""
        readings = pd.DataFrame([
            ('s1', 0, 'Carbon dioxide', 500.),
            ('s1', 0, 'Carbon dioxide', 500.),
            ('s1', 0, 'Carbon dioxide', 501.),
            ('s1', 600, 'Carbon dioxide', 510.),
            ('s1', 600, 'Temperature', 22.),
            ('s1', 600, 'Humidity', 45.),
        ], columns=['sensorid', 'timestamp', 'variable', 'value'])
        cleaned, summary = clean_environment_readings(readings)
        self.assertEqual(summary['exact duplicates removed'], 1)
        self.assertEqual(summary['conflicting readings excluded'], 2)
        self.assertEqual(cleaned['co2_ppm'].tolist(), [510.])

    def test_clean_environment_readings_requires_epoch_seconds(self):
        """Treat ISO strings and Timestamp objects as invalid timestamp input."""
        readings = pd.DataFrame([
            ('s1', '2024-03-01T00:00:00Z', 'Carbon dioxide', 500.),
            ('s1', pd.Timestamp('2024-03-01T00:10:00Z'), 'Temperature', 22.),
            ('s1', 0, 'Carbon dioxide', 510.),
        ], columns=['sensorid', 'timestamp', 'variable', 'value'])
        cleaned, summary = clean_environment_readings(readings)
        self.assertEqual(summary['invalid readings'], 2)
        self.assertEqual(cleaned['co2_ppm'].tolist(), [510.])

    def test_forecast_features_use_backward_lags_and_tolerant_targets(self):
        """Match historical values backward and leave unsupported targets missing."""
        times = pd.to_datetime([
            '2024-03-01T00:00:00Z', '2024-03-01T00:10:00Z',
            '2024-03-01T00:20:00Z', '2024-03-01T00:40:00Z',
        ])
        sensor = pd.DataFrame({
            'co2_ppm': [400., 410., 420., 440.],
            'Temperature': [22.] * 4,
            'Humidity': [45.] * 4,
        }, index=times)
        features = build_forecast_features(sensor)
        row = features.loc[features['issue_time'] == times[2]].iloc[0]
        self.assertEqual(row['co2_lag_10min'], 410.)
        self.assertEqual(row['co2_lag_20min'], 400.)
        self.assertEqual(row['target_co2_ppm'], 440.)
        self.assertEqual(row['co2_rate_ppm_per_min'], 1.)
        unsupported = features.loc[features['issue_time'] == times[1], 'target_co2_ppm'].iloc[0]
        self.assertTrue(pd.isna(unsupported))

    def test_target_boundary_leakage_rejected(self):
        """An overlapping target boundary must fail before any model is fitted."""
        frame = self._make_split('2024-03-01T00:20:00Z', 1)
        with self.assertRaisesRegex(ValueError, 'chronological'):
            compare_models(frame, frame, frame, 'unused_test_output')

    @staticmethod
    def _make_split(start, rows=20):
        """Create a small valid prepared split with independently known times."""
        issue = pd.date_range(start, periods=rows, freq='10min', tz='UTC')
        return pd.DataFrame({
            'issue_time': issue,
            'target_time': issue + pd.Timedelta(minutes=20),
            'lag_time_10': issue - pd.Timedelta(minutes=10),
            'lag_time_20': issue - pd.Timedelta(minutes=20),
            'co2_ppm': np.linspace(400, 450, rows),
            'co2_lag_10min': np.linspace(398, 448, rows),
            'co2_lag_20min': np.linspace(395, 445, rows),
            'Temperature': np.full(rows, 22.),
            'Humidity': np.full(rows, 45.),
            'target_co2_ppm': np.linspace(401, 451, rows),
        })

    def test_successful_model_comparison_exports_results(self):
        """A valid chronological example reaches fitting and writes all outputs."""
        train = self._make_split('2024-03-01T00:00:00Z')
        validation = self._make_split('2024-03-01T06:00:00Z')
        test = self._make_split('2024-03-01T12:00:00Z')
        with tempfile.TemporaryDirectory() as output:
            result = compare_models(train, validation, test, output)
            self.assertEqual(len(result['experiments'].query("split == 'validation'")), 65)
            self.assertIn('test_error_by_week', result)
            for filename in ['model_experiments.csv', 'final_test_metrics.csv', 'test_error_by_week.csv']:
                self.assertTrue((Path(output) / filename).exists())

    def test_test_targets_do_not_change_model_selection(self):
        """Changing withheld targets cannot change validation-based selection."""
        train = self._make_split('2024-03-01T00:00:00Z')
        validation = self._make_split('2024-03-01T06:00:00Z')
        test = self._make_split('2024-03-01T12:00:00Z')
        altered_test = test.copy()
        altered_test['target_co2_ppm'] += 1000
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            original = compare_models(train, validation, test, first)
            altered = compare_models(train, validation, altered_test, second)
            self.assertEqual(original['selected_name'], altered['selected_name'])


if __name__ == '__main__':
    unittest.main()
