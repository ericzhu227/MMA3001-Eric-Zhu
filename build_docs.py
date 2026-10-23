"""Generate pdoc HTML documentation from the current notebook source.

Run ``python build_docs.py`` in the environment containing pdoc (for example,
the Personal conda environment). The notebook's data-processing cells are never
executed during documentation generation.
"""
from pathlib import Path
import ast
import csv
from html import escape
import json


def export_documentation_source(notebook_path, output_path):
    """Create an import-safe documentation module from a Jupyter notebook.

    Args:
        notebook_path (pathlib.Path): UTF-8 notebook containing Python cells.
        output_path (pathlib.Path): Destination for generated Python source.

    Returns:
        pathlib.Path: Generated module containing the notebook walkthrough,
            cell source and top-level function definitions with docstrings.

    Raises:
        OSError: If a source cannot be read or destination cannot be written.
        ValueError: If notebook JSON is invalid.
        SyntaxError: If a notebook code cell is invalid Python.

    Examples:
        Call via ``python build_docs.py`` to keep HTML synchronized with source.
    """
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    narrative = [
        '# CO2 forecasting source documentation',
        'Generated from CO2_Forecasting.ipynb. Edit the notebook, then run '
        '`python build_docs.py` to regenerate this reference.',
        'The current workflow audits environmental measurements, selects one '
        'sensor, builds a 20-minute forecast dataset and evaluates persistence. '
        'Training/validation/test splits are chronological. Linear regression and '
        'decision trees are selected on validation before final test scoring.',
    ]
    functions = []
    for number, cell in enumerate(notebook['cells'], start=1):
        source = ''.join(cell['source'])
        if cell['cell_type'] == 'markdown':
            narrative.append(source)
            continue
        if cell['cell_type'] != 'code':
            continue
        tree = ast.parse(source)
        narrative.append(f"## Cell {number}: {cell.get('id', 'code')}")
        narrative.append(ast.get_docstring(tree) or 'Notebook code cell.')
        narrative.append('```python\n' + source + '\n```')
        # Only expose function definitions: importing the reference cannot load
        # datasets, create splits or overwrite exported CSV files.
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(ast.get_source_segment(source, node))
    module = repr('\n\n'.join(narrative)) + '\n\n'
    module += '# Generated reference; edit the notebook instead.\n'
    module += 'from time import perf_counter\nimport numpy as np\nimport pandas as pd\n\n'
    module += '\n\n'.join(functions) + '\n'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(module, encoding='utf-8')
    return output_path


def build_documentation():
    """Generate the pdoc HTML site in docs/ without running notebook cells.

    Returns:
        pathlib.Path: The generated docs/index.html entry point.

    Raises:
        ModuleNotFoundError: If pdoc or documented module dependencies are absent.
        OSError: If source or HTML files cannot be read or written.
        ValueError: If notebook JSON is invalid.
        SyntaxError: If notebook Python source is invalid.

    Examples:
        Run ``python build_docs.py``, then open ``docs/index.html`` in a browser.
    """
    import pdoc

    root = Path(__file__).resolve().parent
    output_dir = root / 'docs'
    source = export_documentation_source(
        root / 'CO2_Forecasting.ipynb',
        output_dir / '_source' / 'co2_forecasting.py',
    )
    # Google-style docstrings become formatted parameters/returns sections;
    # source panels and search are provided by pdoc's own templates.
    pdoc.render.configure(docformat='google', show_source=True, search=True)
    pdoc.pdoc(source, root / 'co2_models.py', root / 'build_docs.py', output_directory=output_dir)
    report = build_project_report(root, output_dir)
    index = output_dir / 'index.html'
    index_text = index.read_text(encoding='utf-8')
    index_text = index_text.replace(
        '<h2>Available Modules</h2>',
        '<h2>Project documentation</h2><p><a href="project_report.html">Assessment report</a></p>'
        '<h2>Available Modules</h2>',
        1,
    )
    index.write_text(index_text, encoding='utf-8')
    return report


def build_project_report(root, output_dir):
    """Build the ordered HTML project report from repository evidence.

    The report is intentionally separate from pdoc's API reference: it gives
    the assessment narrative a stable 12-section order while linking back to
    the generated source documentation and saved figures.
    """
    result_dir = root / 'model_results'

    def read_csv(name):
        path = result_dir / name
        if not path.exists():
            return []
        with path.open(newline='', encoding='utf-8') as handle:
            return list(csv.DictReader(handle))

    def value(rows, key, name, default='—'):
        for row in rows:
            if row.get(key) == name:
                return row.get('value', default)
        return default

    final_metrics = read_csv('final_test_metrics.csv')
    rising_metrics = read_csv('rising_co2_test_metrics.csv')
    noise_metrics = read_csv('validation_noise_sensitivity.csv')

    def table(headers, rows):
        head = ''.join(f'<th>{escape(str(item))}</th>' for item in headers)
        body = ''.join('<tr>' + ''.join(f'<td>{escape(str(item))}</td>' for item in row) + '</tr>' for row in rows)
        return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'

    sections = [
        ('problem', '1. Engineering problem and aims',
         '<p>Indoor CO₂ concentration is a practical proxy for whether ventilation is keeping up with occupancy and other indoor sources. A short-term forecast can provide an early warning before concentration becomes excessive, supporting inspection or ventilation-control decisions.</p>'
         '<p>The project asks: <strong>how accurately can the selected MSI environmental sensor predict CO₂ concentration approximately 20 minutes ahead?</strong> The intended use is decision support for ventilation monitoring, not automatic control or a claim about health thresholds. The scope is one sensor and one contiguous period with measured targets; it does not establish performance across rooms, seasons or buildings.</p>'),
        ('data', '2. Dataset and inputs/outputs',
         '<p>The repository contains an environmental CSV with JSON-packed readings, an occupancy CSV and a sensor-location workbook. The implemented forecasting dataset uses sensor <code>6012002000869</code>, from <code>2024-02-29 06:04:56 UTC</code> to <code>2024-06-13 17:45:44 UTC</code>. CO₂ is measured in ppm; temperature is in °C; humidity uses the source sensor unit. The target is the measured CO₂ value about 20 minutes after the forecast issue time.</p>'
         '<p>Inputs are current and historical CO₂, temperature, humidity and a CO₂ change-rate feature. Occupancy is excluded because the available occupancy-to-room mapping is marked incorrect or unresolved; using it could associate people counts with the wrong environmental sensor and create misleading features.</p>'),
        ('cleaning', '3. Data cleaning and exploration',
         '<p>The workflow extracts measurement times from the environmental JSON rather than database creation times. Exact duplicate measurements are removed, while conflicting values for the same sensor/time/variable are excluded rather than arbitrarily selecting one. Non-finite values and non-positive CO₂ are excluded; other unusual values remain visible for inspection. No interpolation or target filling is used.</p>'
         '<p>A gap greater than 30 minutes starts a new contiguous period. This is long enough to distinguish an outage from the approximately 10-minute sampling pattern while avoiding invented continuity across a substantial break. The selected sensor-period is the longest by timestamp coverage, chosen before modelling and without using forecast errors. The notebook’s coverage audit and plots document the sampling intervals and gaps.</p>'),
        ('features', '4. Feature construction',
         '<p>The model uses the current CO₂ value, CO₂ values approximately 10 and 20 minutes earlier, temperature and humidity, and a CO₂ rate feature. Historical values are obtained by backward matching: for an issue time, only measurements at or before that time are eligible, with a limited staleness tolerance. This prevents a later reading from leaking into an earlier feature.</p>'
         '<p>Rates are calculated from the actual elapsed time between the matched measurements, rather than assuming every interval is exactly 10 or 20 minutes. The target is also matched to an observed measurement within the declared 20-minute tolerance, so the evaluation reflects available observations rather than interpolated ground truth.</p>'),
        ('validation', '5. Validation design',
         '<p>Forecast issue times are split chronologically into approximately 60% training, 20% validation and 20% final test. Random splitting would allow nearby records from the future to appear in training and would make the evaluation look easier than deployment on genuinely later data.</p>'
         '<p>Boundary purging removes earlier examples whose future target reaches into the next split. The final test period remains untouched during feature-set and model selection. Learned transformations, if used, must be fitted on training data only.</p>'),
        ('models', '6. Model development',
         '<p>Persistence predicts that future CO₂ will equal the current CO₂ value, giving a transparent operational baseline. Ordinary least squares estimates a weighted combination of the selected features. Decision trees partition the feature space into regions and predict the average target in each leaf; depths 2, 4, 6, 8 and unlimited, with minimum leaf sizes 5, 20 and 50, were compared.</p>'
         '<p>Four feature sets compare CO₂ history, environmental variables, and 10- or 20-minute rates. Validation MAE selects the configuration, with persistence included as a candidate, before the frozen selected model is scored on the final test.</p>'),
        ('results', '7. Results',
         '<p>The final test comparison below is generated from the saved model results. Lower MAE and RMSE are better; signed error shows average bias, so it should not be interpreted as accuracy on its own.</p>'
         + table(['Candidate', 'MAE (ppm)', 'RMSE (ppm)', 'Mean signed error (ppm)'], [
             [row.get('candidate'), f"{float(row.get('MAE_ppm')):.4f}", f"{float(row.get('RMSE_ppm')):.4f}", f"{float(row.get('mean_signed_error_ppm')):.4f}"] for row in final_metrics
         ])
         + '<p>The selected linear model with environmental history and the 20-minute rate has final-test MAE 7.8328 ppm and RMSE 9.7250 ppm, compared with persistence at 8.6906 ppm and 10.9208 ppm. The saved figures show the validation comparison and test diagnostics.</p>'
         + '<p><img src="../model_results/validation_comparison.png" alt="Validation model comparison"><img src="../model_results/test_diagnostics.png" alt="Final test diagnostics"></p>'),
        ('sensitivity', '8. Sensitivity and performance',
         '<p>The selected model is only modestly better than persistence, which indicates that recent CO₂ history carries most of the predictive signal. Adding temperature, humidity and rate features provides a small validation improvement. Synthetic input noise increases validation MAE from 10.2028 ppm at 0 ppm noise to 12.0657 ppm at 10 ppm noise, showing sensitivity to measurement quality.</p>'
         + table(['Input noise standard deviation (ppm)', 'MAE (ppm)', 'RMSE (ppm)'], [[row.get('input_noise_std_ppm'), f"{float(row.get('MAE_ppm')):.4f}", f"{float(row.get('RMSE_ppm')):.4f}"] for row in noise_metrics])
         + '<p>On the rising-CO₂ subset, the selected model achieves MAE 8.5846 ppm versus 10.3855 ppm for persistence. Runtime measurements are single-run wall-clock observations rather than formal benchmarks; the parsing and prediction timings should be treated as engineering evidence, not universal performance claims.</p>'),
        ('discussion', '9. Engineering discussion and limitations',
         '<p>The results support using the model as a monitoring aid: it can improve average 20-minute forecasts over persistence for this sensor-period, especially during rising CO₂. It should not be used as the sole basis for ventilation action because the study does not calibrate sensor accuracy, infer causal ventilation effects or evaluate health thresholds.</p>'
         '<p>Errors are expected around abrupt changes, unusual operating conditions and periods unlike the training data. Important limitations are the single sensor, one contiguous period, unresolved room mapping, measured-target tolerance, synthetic rather than calibrated noise, and selection based on one validation period. Decision trees also cannot extrapolate beyond the target values represented in their leaves.</p>'),
        ('conclusion', '10. Conclusion',
         '<p>For the selected MSI sensor and 20-minute measured horizon, the linear model using CO₂ history, environmental history and the 20-minute rate is the strongest tested approach. It reduces final-test MAE from 8.6906 ppm for persistence to 7.8328 ppm and RMSE from 10.9208 ppm to 9.7250 ppm. The improvement is useful but moderate, so the answer is that short-term forecasting is feasible as supplementary ventilation monitoring rather than a standalone control system.</p>'
         '<p>Realistic improvements include resolving the room mapping, evaluating multiple sensors and seasons, collecting higher-frequency measurements, calibrating uncertainty against sensor specifications, adding validated occupancy or calendar features, and using rolling-origin evaluation across more periods.</p>'),
        ('ai', '11. AI-use reflection and references',
         '<p>AI assistance in this repository supported dataset-loading code, JSON extraction, audit logic, chronological splitting, baseline/model implementation, debugging and documentation generation. The author should supplement this statement with their own contribution estimate, examples of accepted or rejected suggestions, and a reflection on how the code and conclusions were verified. Verification evidence includes unit tests, doctest checks, notebook review, saved diagnostics and the test report.</p>'
         '<p>References to complete before submission: the supplied MSI occupancy/environmental dataset and sensor-location workbook; the MMA3001 project brief and teaching notes; and any external sources used for ventilation, CO₂ monitoring, forecasting or model methodology. The repository’s <a href="../README.md">README</a>, <a href="../TEST_REPORT.md">test report</a> and <a href="../PROJECT_CRITERIA.md">criteria checklist</a> record supporting project evidence.</p>'),
        ('submission', '12. Repository and submission materials',
         '<p>The repository includes the executable notebook, model module, tests, generated model results, licence, README, test report and this HTML report. Run <code>python build_docs.py</code> from the repository root to regenerate both the pdoc source documentation and this report. Run <code>python -m unittest discover -s tests -v</code> and <code>python -m doctest co2_models.py</code> to repeat verification.</p>'
         '<p>Before final submission, check that the notebook runs from top to bottom in the declared environment, generated files are included as required, the README lists dependencies and commands, the report contains actual references and AI reflection, and the repository link, zipped clone, written report and presentation materials meet the course instructions.</p>'),
    ]

    nav = ''.join(f'<li><a href="#{anchor}">{escape(title)}</a></li>' for anchor, title, _ in sections)
    content = ''.join(f'<section id="{anchor}"><h2>{escape(title)}</h2>{body}</section>' for anchor, title, body in sections)
    html = f'''<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>MMA3001 CO₂ Forecasting Project Report</title>
<style>
:root {{ color-scheme: light; --ink:#17212b; --muted:#52606d; --accent:#166534; --panel:#f4f7f5; --line:#d8e0db; }}
* {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); font:16px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif; background:#fff; }}
header {{ background:linear-gradient(135deg,#12372a,#236b45); color:white; padding:3.5rem max(1.25rem,calc((100vw - 1100px)/2)); }}
header h1 {{ max-width:850px; margin:0 0 .6rem; font-size:clamp(2rem,5vw,3.5rem); line-height:1.1; }} header p {{ max-width:760px; margin:0; color:#e6f4eb; }}
.layout {{ display:grid; grid-template-columns:250px minmax(0,850px); gap:2.5rem; max-width:1100px; margin:0 auto; padding:2rem 1.25rem 5rem; }}
nav {{ position:sticky; top:1rem; align-self:start; background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:1rem; }} nav strong {{ display:block; margin-bottom:.5rem; }} nav ul {{ list-style:none; padding:0; margin:0; }} nav li {{ margin:.25rem 0; }} nav a {{ color:var(--accent); text-decoration:none; }} nav a:hover {{ text-decoration:underline; }}
section {{ border-bottom:1px solid var(--line); padding:0 0 2rem; margin-bottom:2rem; }} h2 {{ color:var(--accent); line-height:1.2; margin-top:0; }} h3 {{ margin-bottom:.25rem; }} code {{ background:#eef2ef; border-radius:4px; padding:.1rem .3rem; }} .table-wrap {{ overflow-x:auto; }} table {{ border-collapse:collapse; width:100%; margin:1rem 0; font-size:.95rem; }} th,td {{ border:1px solid var(--line); padding:.55rem .65rem; text-align:left; }} th {{ background:var(--panel); }} img {{ max-width:100%; width:48%; margin:.5rem 1% .5rem 0; border:1px solid var(--line); border-radius:6px; }}
@media (max-width:760px) {{ .layout {{ display:block; }} nav {{ position:static; margin-bottom:2rem; }} img {{ width:100%; margin-right:0; }} }}
</style></head>
<body><header><h1>MMA3001: Short-Term Indoor CO₂ Forecasting</h1><p>Ordered project documentation for engineering motivation, data, modelling, validation, results, limitations and submission readiness.</p></header>
<div class="layout"><nav><strong>Contents</strong><ol>{nav}</ol><p><a href="index.html">Code documentation</a></p></nav><main>{content}</main></div></body></html>'''
    report_path = output_dir / 'project_report.html'
    report_path.write_text(html, encoding='utf-8')
    return report_path


if __name__ == '__main__':
    print(build_documentation())
