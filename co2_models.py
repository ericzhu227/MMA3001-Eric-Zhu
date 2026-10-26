"""Compare indoor CO2 forecasts using chronological prepared datasets.

Models are selected by validation MAE only. The final test is scored after
selection, without further tuning. All CO2 errors are in ppm. Timing is wall
clock CPU time and varies with hardware. No scaling or imputation is learned
from validation or test data.
"""
from pathlib import Path
from time import perf_counter
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


def extract_environment_measurements(json_text, sensorid=None):
    """Extract selected measurements from one environmental JSON payload.

    Args:
        json_text (str): JSON payload containing variable/value entries.
        sensorid (str, optional): Sensor identifier copied to each result.

    Returns:
        list of tuple: ``(sensorid, epoch_seconds, variable, value)`` rows for
        CO2, temperature and humidity measurements.

    Raises:
        ValueError: If the payload is not valid JSON.
        TypeError: If the payload is not a JSON list of measurement objects.
    """
    payload = json.loads(json_text)
    if not isinstance(payload, list):
        raise TypeError("Environmental JSON must contain a list.")
    selected = {"Carbon dioxide", "Temperature", "Humidity"}
    records = []
    for item in payload:
        if not isinstance(item, dict):
            raise TypeError("Each environmental entry must be an object.")
        variable = item.get("variable", {})
        if not isinstance(variable, dict):
            raise TypeError("Environmental variable metadata must be an object.")
        name = variable.get("name")
        if name not in selected:
            continue
        values = item.get("values", [])
        if not isinstance(values, list):
            raise TypeError("Environmental values must be a list.")
        for reading in values:
            if not isinstance(reading, dict):
                raise TypeError("Each environmental reading must be an object.")
            records.append((sensorid, reading.get("time"), name, reading.get("value")))
    return records


def clean_environment_readings(readings):
    """Clean extracted environmental readings and return CO2-ready data.

    Args:
        readings (pandas.DataFrame): Long-form readings with ``sensorid``,
            ``timestamp``, ``variable`` and ``value`` columns. ``timestamp``
            values must be numeric Unix epoch seconds; nonnumeric timestamps
            are counted as invalid and excluded.

    Returns:
        tuple: A wide CO2/temperature/humidity DataFrame and a dictionary of
            counts for invalid rows, exact duplicates, conflicts and
            non-positive CO2 values.

    Raises:
        KeyError: If a required long-form column is absent.
    """
    required = {'sensorid', 'timestamp', 'variable', 'value'}
    missing = sorted(required.difference(readings.columns))
    if missing:
        raise KeyError(f'Readings are missing required columns: {missing}')
    frame = readings[list(required)].copy()
    frame['timestamp'] = pd.to_datetime(
        pd.to_numeric(frame['timestamp'], errors='coerce'),
        unit='s', utc=True, errors='coerce'
    )
    frame['value'] = pd.to_numeric(frame['value'], errors='coerce')
    invalid = frame[['sensorid', 'timestamp', 'variable', 'value']].isna().any(axis=1)
    invalid |= ~np.isfinite(frame['value'].fillna(np.nan))
    invalid_count = int(invalid.sum())
    frame = frame.loc[~invalid].copy()
    exact_duplicates = int(frame.duplicated().sum())
    frame = frame.drop_duplicates()
    keys = ['sensorid', 'timestamp', 'variable']
    conflicts = frame.duplicated(keys, keep=False)
    conflicting_count = int(conflicts.sum())
    frame = frame.loc[~conflicts]
    wide = frame.pivot(
        index=['sensorid', 'timestamp'], columns='variable', values='value'
    ).reset_index()
    co2 = wide.rename(columns={'Carbon dioxide': 'co2_ppm'}).sort_values(
        ['sensorid', 'timestamp']
    )
    nonpositive_count = int(co2['co2_ppm'].le(0).sum()) if 'co2_ppm' in co2 else 0
    if 'co2_ppm' in co2:
        co2 = co2.loc[co2['co2_ppm'].gt(0)].copy()
    return co2, {
        'invalid readings': invalid_count,
        'exact duplicates removed': exact_duplicates,
        'conflicting readings excluded': conflicting_count,
        'nonpositive CO2 excluded': nonpositive_count,
    }


def build_forecast_features(sensor_series, horizon='20min', tolerance='2min'):
    """Build backward-matched historical features and measured targets.

    Args:
        sensor_series (pandas.DataFrame): Timestamp-indexed sensor data with
            ``co2_ppm``, ``Temperature`` and ``Humidity`` columns.
        horizon (str or pandas.Timedelta): Desired forecast horizon.
        tolerance (str or pandas.Timedelta): Maximum backward/target matching
            distance; no interpolation is performed.

    Returns:
        pandas.DataFrame: Issue times, backward CO2 lags, elapsed-time rate,
            desired target times and nearest measured targets.

    Raises:
        KeyError: If the sensor data lacks a required column.
        ValueError: If timestamps are invalid or the sensor data is empty.
    """
    required = {'co2_ppm', 'Temperature', 'Humidity'}
    missing = sorted(required.difference(sensor_series.columns))
    if missing:
        raise KeyError(f'Sensor data is missing required columns: {missing}')
    if sensor_series.empty:
        raise ValueError('Sensor data must not be empty.')
    horizon = pd.Timedelta(horizon)
    tolerance = pd.Timedelta(tolerance)
    base = sensor_series[['co2_ppm', 'Temperature', 'Humidity']].reset_index()
    time_column = base.columns[0]
    base = base.rename(columns={time_column: 'timestamp'})
    base['timestamp'] = pd.to_datetime(base['timestamp'], utc=True, errors='coerce')
    if base['timestamp'].isna().any():
        raise ValueError('Sensor data contains invalid timestamps.')
    base = base.sort_values('timestamp')
    features = base.rename(columns={'timestamp': 'issue_time'}).copy()
    for minutes in [10, 20]:
        lookup = base[['timestamp', 'co2_ppm']].rename(columns={
            'timestamp': f'lag_time_{minutes}',
            'co2_ppm': f'co2_lag_{minutes}min',
        })
        features['lookup_time'] = features['issue_time'] - pd.Timedelta(minutes=minutes)
        features = pd.merge_asof(
            features.sort_values('lookup_time'), lookup,
            left_on='lookup_time', right_on=f'lag_time_{minutes}',
            direction='backward', tolerance=tolerance,
        )
    features = features.drop(columns='lookup_time').sort_values('issue_time')
    elapsed = (
        features['issue_time'] - features['lag_time_10']
    ).dt.total_seconds() / 60
    features['co2_rate_ppm_per_min'] = (
        features['co2_ppm'] - features['co2_lag_10min']
    ) / elapsed
    features['desired_target_time'] = features['issue_time'] + horizon
    targets = base[['timestamp', 'co2_ppm']].rename(columns={
        'timestamp': 'target_time', 'co2_ppm': 'target_co2_ppm'
    })
    return pd.merge_asof(
        features.sort_values('desired_target_time'), targets,
        left_on='desired_target_time', right_on='target_time',
        direction='nearest', tolerance=tolerance,
    )


def forecast_metrics(observed, predicted):
    """Calculate forecast errors using prediction minus observation.

    Args:
        observed (array-like): Finite measured CO2 concentrations in ppm.
        predicted (array-like): Same-length finite predictions in ppm.

    Returns:
        dict: MAE_ppm, RMSE_ppm and mean_signed_error_ppm.

    Raises:
        ValueError: If inputs are empty, nonfinite or have unequal shapes.

    Examples:
        >>> forecast_metrics([410, 410], [400, 420])['MAE_ppm']
        10.0
    """
    actual, estimate = np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float)
    if actual.size == 0 or actual.shape != estimate.shape or not (np.isfinite(actual).all() and np.isfinite(estimate).all()):
        raise ValueError('Metrics require nonempty, equal-shaped finite inputs.')
    error = estimate - actual
    return {'MAE_ppm': float(np.abs(error).mean()),
            'RMSE_ppm': float(np.sqrt(np.mean(error**2))),
            'mean_signed_error_ppm': float(error.mean())}


def add_rate_features(frame):
    """Add 10- and 20-minute backward differences using actual times.

    Args:
        frame (pandas.DataFrame): Prepared split with issue_time, lag_time_20,
            co2_ppm and co2_lag_20min. Times must include timezone information.

    Returns:
        pandas.DataFrame: Copy with co2_rate_20_ppm_per_min added.

    Raises:
        KeyError: If required columns are absent.
        ValueError: If lag elapsed times are invalid or nonpositive.
    """
    result = frame.copy()
    elapsed = (pd.to_datetime(result['issue_time'], utc=True) - pd.to_datetime(result['lag_time_20'], utc=True)).dt.total_seconds() / 60
    if elapsed.isna().any() or (elapsed <= 0).any():
        raise ValueError('Backward rates require valid historical timestamps.')
    if {'lag_time_10', 'co2_lag_10min'}.issubset(result.columns):
        elapsed_10 = (pd.to_datetime(result['issue_time'], utc=True) - pd.to_datetime(result['lag_time_10'], utc=True)).dt.total_seconds() / 60
        if elapsed_10.isna().any() or (elapsed_10 <= 0).any():
            raise ValueError('Backward rates require valid historical timestamps.')
        result['co2_rate_ppm_per_min'] = (result.co2_ppm - result.co2_lag_10min) / elapsed_10
    result['co2_rate_20_ppm_per_min'] = (result.co2_ppm - result.co2_lag_20min) / elapsed
    return result


def validate_prepared_split(frame, name='split'):
    """Validate a prepared chronological split before model fitting.

    Args:
        frame (pandas.DataFrame): Prepared rows containing issue/target times,
            backward lag times and finite model inputs in ppm or source units.
        name (str): Label used in validation error messages.

    Returns:
        None: The function returns only when all structural checks pass.

    Raises:
        KeyError: If a required prepared-data column is missing.
        ValueError: If the split is empty, contains invalid timestamps,
            future-looking lags, invalid horizons or non-finite values.
    """
    required = {
        'issue_time', 'target_time', 'target_co2_ppm', 'co2_ppm',
        'co2_lag_10min', 'co2_lag_20min', 'lag_time_10', 'lag_time_20',
        'Temperature', 'Humidity',
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f'{name} is missing required columns: {missing}')
    if frame.empty:
        raise ValueError(f'{name} must not be empty.')
    times = frame[['issue_time', 'target_time', 'lag_time_10', 'lag_time_20']].apply(
        pd.to_datetime, utc=True, errors='coerce'
    )
    if times.isna().any().any():
        raise ValueError(f'{name} contains invalid timestamps.')
    if (times['target_time'] <= times['issue_time']).any():
        raise ValueError(f'{name} contains a target at or before its issue time.')
    if (times['lag_time_10'] > times['issue_time']).any() or (times['lag_time_20'] > times['issue_time']).any():
        raise ValueError(f'{name} contains historical measurements after its issue time.')
    numeric = frame[['target_co2_ppm', 'co2_ppm', 'co2_lag_10min', 'co2_lag_20min', 'Temperature', 'Humidity']]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f'{name} contains missing or nonfinite numeric values.')


def compare_models(train, validation, test, output_dir):
    """Tune linear/tree forecasts on validation, then score the chosen model.

    Args:
        train (pandas.DataFrame): Earliest prepared chronological split.
        validation (pandas.DataFrame): Later tuning split with measured targets.
        test (pandas.DataFrame): Final withheld split; never used for selection.
        output_dir (str or pathlib.Path): Destination for result CSVs and figures.

    Returns:
        dict: Fitted selected model (None for persistence), selected feature
            names, validation experiments, final test metrics and predictions.

    Raises:
        ValueError: If splits overlap, inputs are nonfinite or metrics invalid.
        KeyError: If a required prepared-data column is missing.
        OSError: If output files cannot be written.

    Notes:
        Uses fixed tree depths and minimum leaf sizes. Rate features are linear
        combinations of CO2 history (approximately so for irregular intervals),
        so they need not improve ordinary linear regression. Decision trees
        cannot extrapolate beyond training leaf averages. Repeated runs reuse
        this same test period; do not tune in response to its results.
    """
    train, validation, test = [add_rate_features(p) for p in (train, validation, test)]
    for label, part in [('train', train), ('validation', validation), ('test', test)]:
        validate_prepared_split(part, label)
    for earlier, later in [(train, validation), (validation, test)]:
        if pd.to_datetime(earlier.target_time, utc=True).max() >= pd.to_datetime(later.issue_time, utc=True).min():
            raise ValueError('Splits must be chronological with target-boundary purging.')
    history = ['co2_ppm', 'co2_lag_10min', 'co2_lag_20min']
    environment = history + ['Temperature', 'Humidity']
    feature_sets = {'history': history, 'history_environment': environment,
                    'history_environment_rate10': environment + ['co2_rate_ppm_per_min'],
                    'history_environment_rate20': environment + ['co2_rate_20_ppm_per_min']}
    experiments, candidates = [], {}

    def record(name, kind, columns, model, depth=None, leaf=None):
        """Record a candidate's training/validation errors and runtime only."""
        started = perf_counter()
        if model is not None:
            model.fit(train[columns], train.target_co2_ppm)
        fit_seconds = perf_counter() - started
        for label, part in [('train', train), ('validation', validation)]:
            started = perf_counter()
            prediction = part.co2_ppm.to_numpy(copy=True) if model is None else model.predict(part[columns])
            seconds = perf_counter() - started
            experiments.append({'candidate': name, 'model': kind, 'features': ', '.join(columns),
                'max_depth': depth, 'min_samples_leaf': leaf, 'split': label,
                **forecast_metrics(part.target_co2_ppm, prediction),
                'fit_seconds': fit_seconds, 'prediction_seconds': seconds})
        candidates[name] = (model, columns)

    record('persistence', 'persistence', ['co2_ppm'], None)
    for feature_name, columns in feature_sets.items():
        record('linear_' + feature_name, 'linear', columns, LinearRegression())
        for depth in [2, 4, 6, 8, None]:
            for leaf in [5, 20, 50]:
                name = f'tree_{feature_name}_depth{depth}_leaf{leaf}'
                record(name, 'tree', columns, DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf, random_state=42), depth, leaf)
    results = pd.DataFrame(experiments)
    ranking = results.loc[results.split == 'validation'].sort_values(['MAE_ppm', 'candidate']).reset_index(drop=True)
    selected_name = ranking.iloc[0]['candidate']
    model, columns = candidates[selected_name]
    # Freeze selection before touching final test targets. Retain the training-only fit.
    summaries, predictions = [], pd.DataFrame({'issue_time': test.issue_time, 'target_time': test.target_time, 'observed_co2_ppm': test.target_co2_ppm})
    for name, estimator, cols in [('persistence', None, ['co2_ppm']), (selected_name, model, columns)]:
        if name in predictions.columns:
            continue
        started = perf_counter()
        pred = test.co2_ppm.to_numpy(copy=True) if estimator is None else estimator.predict(test[cols])
        seconds = perf_counter() - started
        predictions[name] = pred
        summaries.append({'candidate': name, **forecast_metrics(test.target_co2_ppm, pred), 'prediction_seconds': seconds})
    # Compare the frozen models over calendar weeks to show whether the test
    # improvement is consistent rather than driven by one short interval.
    temporal_errors = pd.DataFrame({
        'target_time': pd.to_datetime(predictions.target_time, utc=True),
        'persistence_error_ppm': predictions.persistence - predictions.observed_co2_ppm,
        'selected_error_ppm': predictions[selected_name] - predictions.observed_co2_ppm,
    })
    temporal_errors['target_week'] = temporal_errors['target_time'].dt.strftime('%Y-W%U')
    weekly_errors = temporal_errors.groupby('target_week', sort=True).agg(
        rows=('target_time', 'size'),
        persistence_mae_ppm=('persistence_error_ppm', lambda values: np.abs(values).mean()),
        selected_mae_ppm=('selected_error_ppm', lambda values: np.abs(values).mean()),
    ).reset_index()
    weekly_errors['improvement_ppm'] = (
        weekly_errors['persistence_mae_ppm'] - weekly_errors['selected_mae_ppm']
    )
    weekly_errors['improvement_percent'] = np.where(
        weekly_errors['persistence_mae_ppm'].ne(0),
        100 * weekly_errors['improvement_ppm'] / weekly_errors['persistence_mae_ppm'],
        np.nan,
    )

    # Report rising periods using an a-priori historical rate threshold, never a target-based filter.
    rising = test.co2_rate_ppm_per_min > 1.0
    rising_rows = []
    if rising.any():
        for row in summaries:
            name = row['candidate']
            rising_rows.append({'candidate': name, 'rows': int(rising.sum()), **forecast_metrics(test.loc[rising, 'target_co2_ppm'], predictions.loc[rising, name])})
    # Stress only input features at validation time; this is synthetic sensitivity,
    # not a new accuracy estimate or a basis for retuning after test evaluation.
    noise_rows = []
    rng = np.random.default_rng(42)
    for sigma in [0., 5., 10.]:
        noisy = validation.copy()
        noisy[history] = noisy[history] + rng.normal(0, sigma, (len(noisy), len(history)))
        elapsed = (pd.to_datetime(noisy.issue_time, utc=True) - pd.to_datetime(noisy.lag_time_10, utc=True)).dt.total_seconds() / 60
        noisy['co2_rate_ppm_per_min'] = (noisy.co2_ppm - noisy.co2_lag_10min) / elapsed
        noisy = add_rate_features(noisy)
        pred = noisy.co2_ppm.to_numpy() if model is None else model.predict(noisy[columns])
        noise_rows.append({'input_noise_std_ppm': sigma, **forecast_metrics(noisy.target_co2_ppm, pred)})
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / 'model_experiments.csv', index=False)
    ranking.to_csv(output / 'validation_ranking.csv', index=False)
    pd.DataFrame(summaries).to_csv(output / 'final_test_metrics.csv', index=False)
    predictions.to_csv(output / 'final_test_predictions.csv', index=False)
    weekly_errors.to_csv(output / 'test_error_by_week.csv', index=False)
    pd.DataFrame(rising_rows).to_csv(output / 'rising_co2_test_metrics.csv', index=False)
    pd.DataFrame(noise_rows).to_csv(output / 'validation_noise_sensitivity.csv', index=False)
    if hasattr(model, 'coef_'):
        pd.DataFrame({'feature': columns, 'coefficient': model.coef_}).to_csv(output / 'selected_model_coefficients.csv', index=False)
    elif hasattr(model, 'feature_importances_'):
        pd.DataFrame({'feature': columns, 'importance': model.feature_importances_}).to_csv(output / 'selected_model_importance.csv', index=False)
    return {'selected_name': selected_name, 'model': model, 'features': columns,
            'experiments': results, 'ranking': ranking, 'test_metrics': pd.DataFrame(summaries),
            'test_predictions': predictions, 'noise_sensitivity': pd.DataFrame(noise_rows),
            'test_error_by_week': weekly_errors}


def plot_model_results(comparison, output_dir):
    """Save and display validation comparisons and final test diagnostic plots.

    Args:
        comparison (dict): Results returned by compare_models.
        output_dir (str or pathlib.Path): Folder for PNG figures.

    Returns:
        None: Saves validation_comparison.png and test_diagnostics.png.

    Raises:
        ImportError: If matplotlib is not installed.
        OSError: If figures cannot be saved.
    """
    import matplotlib.pyplot as plt
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best = comparison['ranking'].groupby('model', sort=False).head(1).copy()
    best['family_label'] = best['model'].replace({
        'persistence': 'Persistence',
        'linear': 'Linear',
        'tree': 'Decision tree',
    })
    best_value = best['MAE_ppm'].min()
    best['excess_ppm'] = best['MAE_ppm'] - best_value

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.5), gridspec_kw={'width_ratios': [1.2, 1]}
    )

    # A tight y-axis and labels make small absolute differences visible.
    bars = axes[0].bar(
        best['family_label'], best['MAE_ppm'],
        color=['#4c78a8', '#59a14f', '#f28e2b'],
    )
    value_range = max(best['MAE_ppm'].max() - best['MAE_ppm'].min(), 0.1)
    axes[0].set_ylim(
        best['MAE_ppm'].min() - 0.15 * value_range,
        best['MAE_ppm'].max() + 0.25 * value_range,
    )
    axes[0].bar_label(bars, fmt='%.3f ppm', padding=3, fontsize=9)
    axes[0].set_ylabel('Validation MAE (ppm)')
    axes[0].set_title('Absolute error (zoomed scale)')
    axes[0].grid(axis='y', alpha=.25)

    # A zero-based difference view directly shows the ranking gaps.
    bars = axes[1].bar(best['family_label'], best['excess_ppm'], color='#9c755f')
    axes[1].bar_label(bars, fmt='%.3f ppm', padding=3, fontsize=9)
    axes[1].set_ylim(0, max(best['excess_ppm'].max() * 1.35, 0.1))
    axes[1].set_ylabel('Extra MAE above best family (ppm)')
    axes[1].set_title('Difference from best configuration')
    axes[1].grid(axis='y', alpha=.25)

    for axis in axes:
        axis.tick_params(axis='x', rotation=15)
    fig.suptitle('Best validation configuration per model family')
    fig.tight_layout()
    fig.savefig(output / 'validation_comparison.png', dpi=160)
    plt.show()
    predictions = comparison['test_predictions']
    name = comparison['selected_name']
    error = predictions[name] - predictions.observed_co2_ppm
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    import matplotlib.dates as mdates
    times = pd.to_datetime(predictions.target_time, utc=True)
    axes[0].plot(times, predictions.observed_co2_ppm, label='Measured', linewidth=.8)
    axes[0].plot(times, predictions[name], label='Selected forecast', linewidth=.8, alpha=.7)
    axes[0].set(ylabel='CO2 (ppm)', title='Final test: measured and forecast'); axes[0].legend()
    axes[1].plot(times, error, linewidth=.6); axes[1].axhline(0, color='black', linewidth=.8)
    axes[1].set(ylabel='Prediction minus measured (ppm)', xlabel='Target time (UTC)')
    for ax in axes[:2]:
        locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axes[2].hist(error, bins=40); axes[2].set(xlabel='Residual (ppm)', ylabel='Count')
    fig.tight_layout(); fig.savefig(output / 'test_diagnostics.png', dpi=160)
    plt.show()
