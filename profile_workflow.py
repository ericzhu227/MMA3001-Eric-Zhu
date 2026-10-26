"""Profile the prepared-data model-comparison workflow.

This is separate from the notebook so profiling can be repeated without
parsing the raw dataset. It writes a cProfile report and a fresh comparison
under ``model_results/profiled_comparison``.
"""
from cProfile import Profile
from pathlib import Path
from pstats import SortKey, Stats

import pandas as pd

from co2_models import compare_models


def main():
    """Run cProfile around model comparison using the prepared CSV splits."""
    root = Path(__file__).resolve().parent
    prepared = root / 'prepared_data'
    output = root / 'model_results' / 'profiled_comparison'
    output.mkdir(parents=True, exist_ok=True)
    splits = {
        name: pd.read_csv(prepared / f'{name}.csv')
        for name in ('train', 'validation', 'test')
    }
    profile = Profile()
    profile.enable()
    comparison = compare_models(
        splits['train'], splits['validation'], splits['test'], output
    )
    profile.disable()
    profile_path = output / 'compare_models_profile.txt'
    with profile_path.open('w', encoding='utf-8') as handle:
        Stats(profile, stream=handle).sort_stats(SortKey.CUMULATIVE).print_stats(25)
    print(f"Selected model: {comparison['selected_name']}")
    print(f"Profile written to: {profile_path}")


if __name__ == '__main__':
    main()
