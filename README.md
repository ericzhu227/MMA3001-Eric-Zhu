# MMA3001-Eric-Zhu
Private GitHub for MMA3001

Below was generated using ChatGPT

Responsible AI Usage Statement

Generative artificial intelligence (AI) tools were used to support the preparation of this assessment. AI was used to assist with brainstorming ideas, improving the clarity and grammar of written sections, refining sentence structure, and providing feedback on the organisation and presentation of the report. AI was also used to explain concepts and suggest approaches to problem-solving where required.

All technical analysis, calculations, design decisions, coding, interpretation of results, and final conclusions were completed, verified, and critically evaluated by the author. AI-generated content was reviewed, modified where necessary, and checked against unit materials, academic literature, and other appropriate sources to ensure accuracy and compliance with academic integrity requirements.

The author accepts full responsibility for the content of this submission and confirms that all submitted work represents their own understanding and learning outcomes.

## Running the CO2 forecasting project

Use Python 3.12 with pandas >= 2.0 and NumPy. Jupyter/ipykernel is needed to run the notebook; openpyxl enables location workbook reading and matplotlib enables optional plots. PyTorch is optional and does not accelerate the pandas workflow.

Keep the supplied files in `MMA3001 Dataset - MSI Occupancy and Environmental`, then run `CO2_Forecasting.ipynb` from top to bottom. Leave `NROWS = None` for the full audit; preview row limits are not suitable for final results. Occupancy remains deferred until the room mapping is resolved. The implemented horizon is 20 minutes, based on the measured sampling intervals. Generated data and audit CSVs are written to `prepared_data/`, replacing earlier generated files.

## Source documentation

Code should include inline comments and docstrings. Use comments to explain intent and consistent Google-style docstrings for functions, including inputs, outputs, exceptions and examples. Follow this convention for future model code.

Open [HTML source documentation](docs/index.html) locally in a browser. It includes every notebook code cell, function docstrings and complete source. Regenerate after edits from the repository folder:

```powershell
python build_docs.py
```

The generator uses pdoc with Google-style docstrings, source panels and search. Run it in the Personal conda environment where pdoc is installed (`conda activate Personal`, then `python build_docs.py`). It does not execute notebook cells or load datasets. The generated `docs/_source/co2_forecasting.py` reference is rebuilt from the notebook; edit the notebook rather than that generated file. Review [project criteria and remaining work](PROJECT_CRITERIA.md) before submission. The HTML source documentation is separate from the required written report.

AI assistance in this repository also includes dataset-loading code, auditing, chronological splitting, baseline implementation, debugging and documentation generation. This supplements the earlier general AI statement; the student still needs to record their own review, understanding, contribution estimate and critical reflection.

## Model comparison

Run the final notebook cells in the Personal environment with scikit-learn and matplotlib installed. `co2_models.py` compares persistence, linear regression and decision trees using CO2 history, temperature/humidity and 10/20-minute backward rate features. Validation MAE selects the configuration before scoring the final test. Do not tune on the exported test results. Tables, predictions and PNG plots are saved under `model_results/`. No serialized model file is required: the fitted estimator is available as `comparison["model"]` in the notebook.

Run verification with `python -m unittest discover -s tests -v` and `python -m doctest co2_models.py`. See `TEST_REPORT.md` for checks and limitations. Regenerate pdoc HTML with `python build_docs.py`.
