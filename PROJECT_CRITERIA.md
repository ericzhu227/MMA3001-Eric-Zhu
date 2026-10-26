# Project criteria and current status

Source: MMA3001_2026_Project_Brief.pdf, especially pages 4-8 and the
documentation rubric on page 12. This checklist records evidence and final
submission actions; it is not a substitute for the written report.

| Criterion | Current evidence / remaining work |
|---|---|
| Engineering problem, inputs and outputs | Indoor CO2 forecasting for ventilation monitoring. Inputs are environmental JSON measurements, UTC measurement times, CO2 in ppm, temperature and source humidity. Outputs are audit tables, timestamped features, measured future targets and chronological split CSVs. Complete the final report and explain supported operating conditions. |
| Reproducible workflow | `requirements.txt` specifies the environment. README gives the authorised raw-data source requirement, exact filenames/folder layout, notebook command and prepared-data reproduction command. |
| Inline comments | Notebook comments explain duplicate handling, backward lag matching, target tolerance, boundary purging and withholding the test set. |
| Consistent docstrings | Public model and preprocessing functions have Google-style arguments, returns and exceptions. `validate_prepared_split()` now follows the same standard. |
| HTML source documentation | `docs/index.html` is the pdoc entry point. `co2_forecasting.html` covers notebook code and the model API. Rebuild with `python build_docs.py` after final edits. |
| Baseline and alternatives | Persistence, linear regression and decision-tree comparisons are implemented and tested. Validation MAE selects the configuration; the final test remains withheld until selection. |
| Validation and verification | Chronological 60/20/20 splitting with target-boundary purging is implemented. Eight unit tests plus the `co2_models.py` doctest cover metrics, JSON validation, duplicate/conflict cleaning, lag/target matching, model exports and leakage protection. |
| Sensitivity and performance | Feature sets, tree depth/leaf-size settings, synthetic input-noise sensitivity and weekly paired errors are exported. `profile_workflow.py` writes cProfile evidence identifying cumulative runtime functions. |
| Engineering limitations | One sensor and period are evaluated; occupancy is excluded because room mapping is unresolved. Approximately 10-minute sampling supports a measured 20-minute forecast, not an exact 15-minute target. No interpolated ground truth or external sensor validation is claimed. |
| Written report | Complete the separate 5-10 page report, references, figures and engineering interpretation. HTML source documentation does not replace the written report. |
| AI disclosure and reflection | README now accurately states that AI assisted code generation, debugging, testing, profiling, documentation and wording. Add the author's contribution estimate, verified/rejected suggestions and personal reflection before submission. |
| Repository and submission | GPL v3 `LICENSE`, `.gitignore`, dependency file, tests, prepared evidence and generated documentation are present. Confirm meaningful GitHub history, working main branch, marker access, final ZIP contents and presentation/demo materials. |
