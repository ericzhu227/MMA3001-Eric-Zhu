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
        variable = item.get("variable", {})
        name = variable.get("name")
        if name not in selected:
            continue
        for reading in item.get("values", []):
            records.append((sensorid, reading.get("time"), name, reading.get("value")))
    return records


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
    """Add a 20-minute backward difference using actual measurement times.

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
    result['co2_rate_20_ppm_per_min'] = (result.co2_ppm - result.co2_lag_20min) / elapsed
    return result


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
    weekly_errors['improvement_percent'] = (
        100 * weekly_errors['improvement_ppm'] / weekly_errors['persistence_mae_ppm']
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
    best = comparison['ranking'].groupby('model', sort=False).head(1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(best.model, best.MAE_ppm)
    ax.set(ylabel='Validation MAE (ppm)', title='Best validation configuration per model family')
    fig.tight_layout(); fig.savefig(output / 'validation_comparison.png', dpi=160)
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
