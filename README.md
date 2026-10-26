# MMA3001-Eric-Zhu

## Project

This project forecasts indoor CO2 concentration approximately 20 minutes ahead
for ventilation monitoring. The workflow compares persistence, linear
regression and decision-tree models using chronological train, validation and
final-test splits. The final test is scored only after validation-based model
selection.

## Environment setup

Use Python 3.10 or newer and install the pinned core dependencies plus the
bounded notebook/plotting dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The tested core versions are Python 3.10.5, NumPy 2.2.6, pandas 2.3.3,
scikit-learn 1.7.2, openpyxl 3.1.5 and pdoc 16.0.0. Matplotlib is required
for PNG figures but is optional for model metrics.

## Raw-data setup

The raw dataset is subject to course access and is not redistributed in this
repository. Obtain it from the authorised MMA3001 course source, then create
this exact folder beside the notebook:

```text
MMA3001 Dataset - MSI Occupancy and Environmental/
├── 5EnvSensor_MayToDec2024_180kRows.csv
├── 5occupancySensor_MayToDec2024_9MRows.csv
└── Sensor ID and Locations.xlsx
```

The notebook searches the current folder and its parents for that folder.
Run `CO2_Forecasting.ipynb` from top to bottom with `NROWS = None` for the
full raw-data workflow. Occupancy remains deferred because the available room
mapping is marked incorrect or unresolved. The notebook writes prepared splits
and audit tables to `prepared_data/` and model results to `model_results/`.

## Reproduce the model comparison from prepared data

If the authorised raw files are unavailable, the included prepared CSVs are
enough to reproduce model fitting and selection:

```powershell
.\.venv\Scripts\python.exe profile_workflow.py
```

The script loads `prepared_data/train.csv`, `validation.csv` and `test.csv`,
runs the same comparison, and writes a cProfile report to
`model_results/profiled_comparison/compare_models_profile.txt`.

## Tests and documentation

Run the tests, doctest and documentation build with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m doctest co2_models.py
.\.venv\Scripts\python.exe build_docs.py
```

Open [HTML source documentation](docs/index.html) locally. The generated
documentation includes notebook code cells, function docstrings and the ordered
project report. Review [PROJECT_CRITERIA.md](PROJECT_CRITERIA.md) and
[TEST_REPORT.md](TEST_REPORT.md) before submission.

## AI-use disclosure

AI assistance was used for code generation and revision suggestions covering
dataset loading, JSON extraction, cleaning audits, chronological feature
construction, model comparison, validation checks, testing, profiling and
HTML documentation. AI also helped improve wording and organisation. The
author reviewed the generated changes, ran the tests/doctest and workflow,
checked the numerical outputs against the prepared datasets and accepts
responsibility for the final implementation, interpretation and conclusions.
The author estimates that AI contributed approximately 70% of the coding and
report drafting assistance, while the author remained responsible for checking
the timestamp series, tracing the workflow, verifying the results, rejecting
unsupported suggestions and making the final engineering decisions. Add any
final personal reflection required by the submission instructions.

## Submission notes

Do not add PyTorch, GPU support, occupancy inputs or a neural network unless a
future project requirement specifically calls for them. Keep the GPL v3
`LICENSE`, meaningful Git history, the final report, presentation/demo
materials and an authorised repository link in the final submission.

Repository: [MMA3001-Eric-Zhu on GitHub](https://github.com/ericzhu227/MMA3001-Eric-Zhu.git)
