# Model verification report

Run date: 22 October 2026 (Australia/Sydney).

Environment: Personal conda environment, scikit-learn 1.7.2.

Commands:

```powershell
python -m unittest discover -s tests -v
python -m doctest co2_models.py
python build_docs.py
```

Result: four unit tests passed, metric docstring example passed. Checks cover hand-computed MAE/RMSE/error sign, invalid metric inputs, irregular elapsed-time rates, and rejection of overlapping target/split boundaries.

Integration run: loaded the existing 4,712 training, 1,569 validation and 1,571 final-test rows; fitted 65 candidates (persistence, four linear configurations and 60 trees). Each candidate was evaluated on train/validation. Selected linear_history_environment_rate20 using validation MAE alone. Saved CSV diagnostics and two PNG plots, then regenerated pdoc HTML.

Validation MAE (ppm): selected linear regression 10.2028; best tree 10.6404; persistence 10.6437. Final-test MAE: selected linear regression 7.8328; persistence 8.6906. Final-test RMSE: selected model 9.7250; persistence 10.9208. Signed mean error: selected model +1.2453 ppm, persistence +0.1763 ppm. Lower average absolute error does not imply lower bias.

Limits: tests verify selected computational contracts, not sensor calibration, room mapping or generalisation to other sensors/seasons. One chronological validation period informs model selection, so validation ranking is optimistic. Test results should not guide further tuning. Noise experiments are synthetic input perturbations and not calibrated sensor uncertainty. Timing is single-run wall-clock timing. Optional accumulated excess-CO2 integration is not implemented.
