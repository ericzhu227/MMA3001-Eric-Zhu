"""Build pdoc source documentation and an ordered HTML project report.

The notebook is never executed during documentation generation. This keeps
the documentation build safe to run without loading the large raw datasets.
"""

from pathlib import Path
import ast
import csv
from html import escape
import json


def export_documentation_source(notebook_path, output_path):
    """Create an import-safe reference module from a Jupyter notebook.

    Args:
        notebook_path (Path): UTF-8 notebook containing Python cells.
        output_path (Path): Destination for generated Python source.

    Returns:
        Path: Generated module containing notebook narrative and functions.

    Raises:
        OSError: If a source cannot be read or written.
        ValueError: If notebook JSON is invalid.
        SyntaxError: If a code cell is invalid Python.
    """
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    narrative = [
        "# CO2 forecasting source documentation",
        "Generated from CO2_Forecasting.ipynb. Edit the notebook, then run "
        "python build_docs.py to regenerate this reference.",
        "The workflow audits environmental measurements, selects one sensor, "
        "builds a 20-minute forecast dataset and evaluates persistence, "
        "linear regression and decision trees using chronological splits.",
    ]
    functions = []

    for number, cell in enumerate(notebook["cells"], start=1):
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            narrative.append(source)
            continue
        if cell["cell_type"] != "code":
            continue

        tree = ast.parse(source)
        narrative.append(f"## Cell {number}: {cell.get('id', 'code')}")
        narrative.append(ast.get_docstring(tree) or "Notebook code cell.")
        code_fence = "`" * 3
        narrative.append(
            "Python source:\n"
            + code_fence
            + "python\n"
            + source
            + "\n"
            + code_fence
        )

        # Only expose function definitions so importing the reference does not
        # load data, create splits or overwrite generated files.
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(ast.get_source_segment(source, node))

    # Add a simple heading before pdoc lists the extracted notebook functions.
    narrative.append("## Functions in notebook")
    module = repr("\n\n".join(narrative)) + "\n\n"
    module += "# Generated reference; edit the notebook instead.\n"
    module += "from time import perf_counter\nimport numpy as np\nimport pandas as pd\n\n"
    module += "\n\n".join(functions) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(module, encoding="utf-8")
    return output_path


def _read_results(result_dir, filename):
    """Read a generated results CSV, returning an empty list if unavailable."""
    path = result_dir / filename
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _make_table(headers, rows):
    """Return an escaped HTML table from string headers and rows."""
    header_html = "".join(f"<th>{escape(str(item))}</th>" for item in headers)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{escape(str(item))}</td>" for item in row) + "</tr>"
        for row in rows
    )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        f"{header_html}</tr></thead><tbody>{body_html}</tbody></table></div>"
    )


def build_project_report(root, output_dir):
    """Build the ordered twelve-section HTML report from repository evidence."""
    result_dir = root / "model_results"
    prepared_dir = root / "prepared_data"
    final_metrics = _read_results(result_dir, "final_test_metrics.csv")
    noise_metrics = _read_results(result_dir, "validation_noise_sensitivity.csv")
    weekly_metrics = _read_results(result_dir, "test_error_by_week.csv")
    exclusion_metrics = _read_results(prepared_dir, "exclusion_diagnostics.csv")

    final_rows = [
        [
            row.get("candidate", ""),
            f"{float(row['MAE_ppm']):.4f}",
            f"{float(row['RMSE_ppm']):.4f}",
            f"{float(row['mean_signed_error_ppm']):.4f}",
        ]
        for row in final_metrics
    ]
    noise_rows = [
        [
            row.get("input_noise_std_ppm", ""),
            f"{float(row['MAE_ppm']):.4f}",
            f"{float(row['RMSE_ppm']):.4f}",
        ]
        for row in noise_metrics
    ]
    weekly_rows = [
        [
            row.get("target_week", ""),
            row.get("rows", ""),
            f"{float(row['persistence_mae_ppm']):.4f}",
            f"{float(row['selected_mae_ppm']):.4f}",
            f"{float(row['improvement_ppm']):.4f}",
        ]
        for row in weekly_metrics
    ]
    exclusion_rows = [
        [row.get("condition", ""), row.get("rows", ""), row.get("share_percent", "")]
        for row in exclusion_metrics
    ]

    sections = [
        ("problem", "1. Engineering problem and aims",
         "<p>Indoor CO2 forecasting can provide an early warning that "
         "ventilation may not be keeping pace with indoor sources. The "
         "project asks how accurately the selected MSI sensor can predict "
         "measured CO2 approximately 20 minutes ahead.</p>"),
        ("data", "2. Dataset and inputs/outputs",
         "<p>The environmental CSV contains 179,447 database rows with "
         "JSON-packed readings, while "
         "the occupancy CSV and location workbook provide supporting data. "
         "The model uses CO2 in ppm, temperature, humidity, historical CO2 "
         "and a rate feature. Occupancy is excluded because the available "
         "room mapping is unresolved. The selected sensor-period spans "
         "2024-02-29 06:04:56 to 2024-06-13 17:45:44 UTC, spanning "
         "approximately 105.49 days. Observed CO2 was approximately 403-646 "
         "ppm, with a mean near 473.99 ppm. This range does not establish "
         "performance at much higher concentrations.</p>"),
        ("cleaning", "3. Data cleaning and exploration",
         "<p>Measurement times are extracted from the environmental JSON. "
         "Exact duplicates are removed, conflicting values are excluded, "
         "and non-finite or non-positive CO2 values are removed. A gap over "
         "30 minutes starts a new contiguous period; selection uses time "
         "coverage rather than forecast performance. Of 13,800 selected-sensor "
         "issue times, 7,855 become usable forecasts before boundary purging. "
         "Approximately 43.1% of candidates are excluded, and 409,812 "
         "duplicate individual readings (not CSV rows) were removed. The "
         "exclusion categories below overlap, so they must not be added "
         "as mutually exclusive totals.</p>"
         + _make_table(
             ["Exclusion condition", "Rows", "Share of candidate rows (%)"],
             exclusion_rows,
         )),
        ("features", "4. Feature construction",
         "<p>Features include current CO2, CO2 values approximately 10 and "
         "20 minutes earlier, temperature, humidity and CO2 rate. Historical "
         "values use backward matching so later readings cannot leak into "
         "earlier inputs. Rates use actual elapsed time.</p>"),
        ("validation", "5. Validation design",
         "<p>Issue times are split chronologically into 4,712 training, "
         "1,569 validation and 1,571 final-test rows. Boundary purging "
         "removes earlier examples whose target reaches into the next "
         "period, protecting the final test from tuning.</p>"),
        ("models", "6. Model development",
         "<p>Persistence provides a transparent baseline. Ordinary least "
         "squares estimates a weighted combination of features. Decision "
         "trees partition the feature space and predict the average target "
         "in each leaf. The comparison contains 65 candidates: persistence, "
         "four linear configurations and 60 tree configurations. Validation "
         "MAE selects the configuration.</p>"),
        ("results", "7. Results",
         "<p>Lower MAE and RMSE indicate smaller errors; signed error is "
         "reported separately as a bias measure.</p>"
         + _make_table(
             ["Candidate", "MAE (ppm)", "RMSE (ppm)", "Signed error (ppm)"],
             final_rows,
         )
         + '<p><img src="../model_results/validation_comparison.png" '
         'alt="Validation model comparison"><img '
         'src="../model_results/test_diagnostics.png" '
         'alt="Final test diagnostics"></p>'
         "<p>The selected model improves test MAE from 8.6906 ppm for persistence to 7.8328 ppm, a reduction of 0.8579 ppm or approximately 9.9%. Its positive signed error is 1.2453 ppm, compared with 0.1763 ppm for persistence, so the lower average error comes with slightly greater positive bias.</p>"
         + _make_table(
             ["Target week", "Rows", "Persistence MAE", "Selected MAE", "Improvement"],
             weekly_rows,
         )),
        ("sensitivity", "8. Sensitivity and performance",
         "<p>The selected model is only modestly better than persistence, "
         "indicating that recent CO2 history carries most of the signal. "
         "Synthetic input noise increases validation error, while timing "
         "results should be treated as single-run engineering evidence. "
         "The selected linear fit took approximately 0.00147 seconds and "
         "the final test contains 1,571 rows; these are not formal benchmarks.</p>"
         + _make_table(
             ["Input noise standard deviation (ppm)", "MAE (ppm)", "RMSE (ppm)"],
             noise_rows,
         )
         + "<p>The environmental and rate features change validation MAE only slightly relative to CO2 history. The 20-minute rate is largely derived from current and lagged CO2, so the evidence does not justify claiming a substantial independent contribution from the extra features. The existing selection is retained because it was made by validation MAE, but the complexity benefit is weak.</p>"),
        ("discussion", "9. Engineering discussion and limitations",
         "<p>The model is suitable as supplementary monitoring evidence, "
         "not as the sole basis for ventilation action. The study covers "
         "one sensor and one period and does not establish calibration, "
         "causal ventilation effects, uncertainty intervals, useful ventilation "
         "decisions or generalisation to other rooms and seasons. It also does "
         "not establish automatic control or health-threshold performance.</p>"),
        ("conclusion", "10. Conclusion",
         "<p>For this sensor and measured 20-minute horizon, the selected "
         "linear model improves on persistence, but the improvement is "
         "moderate. Future work should resolve room mapping, evaluate more "
         "sensors and seasons, collect higher-frequency data and validate "
         "prediction uncertainty.</p>"),
        ("ai", "11. AI-use reflection and references",
         '<p>OpenAI ChatGPT and Codex assisted with ideas, spelling, code '
         'generation, debugging, report structure and documentation. The '
         'author estimates approximately 70% AI contribution to coding and '
         'drafting assistance, while personally checking the timestamp series, '
         'tracing the workflow, verifying outputs and making final decisions. '
         'References for the dataset and external methods are provided in '
         'the References section of the final submitted PDF. See '
         '<a href="../README.md">README</a>, '
         '<a href="../TEST_REPORT.md">test report</a> and '
         '<a href="../PROJECT_CRITERIA.md">criteria checklist</a>.</p>'),
        ("submission", "12. Reproduction and submission checklist",
         "<p>This HTML page is a supplementary project summary; the PDF is "
         "the submitted report. For reproduction and submission, rerun the "
         "notebook, run the tests and "
         "doctest, regenerate the HTML documentation, check dependencies "
         "and README instructions, and confirm that the report, repository "
         "link, archive and presentation materials are complete.</p>"),
    ]

    navigation = "".join(
        f'<li><a href="#{anchor}">{escape(title)}</a></li>'
        for anchor, title, _ in sections
    )
    content = "".join(
        f'<section id="{anchor}"><h2>{escape(title)}</h2>{body}</section>'
        for anchor, title, body in sections
    )
    document = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MMA3001 CO2 Forecasting Project Report</title>
<style>
:root {{ color-scheme: light; --ink:#17212b; --accent:#166534;
--panel:#f4f7f5; --line:#d8e0db; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); font:16px/1.65 system-ui,
-apple-system,"Segoe UI",sans-serif; background:#fff; }}
header {{ background:linear-gradient(135deg,#12372a,#236b45);
color:white; padding:3.5rem max(1.25rem,calc((100vw - 1100px)/2)); }}
header h1 {{ max-width:850px; margin:0 0 .6rem; font-size:clamp(2rem,5vw,3.5rem);
line-height:1.1; }}
header p {{ max-width:760px; margin:0; color:#e6f4eb; }}
.layout {{ display:grid; grid-template-columns:250px minmax(0,850px);
gap:2.5rem; max-width:1100px; margin:0 auto; padding:2rem 1.25rem 5rem; }}
nav {{ position:sticky; top:1rem; align-self:start; background:var(--panel);
border:1px solid var(--line); border-radius:10px; padding:1rem; }}
nav a {{ color:var(--accent); text-decoration:none; }}
section {{ border-bottom:1px solid var(--line); padding-bottom:2rem;
margin-bottom:2rem; }}
h2 {{ color:var(--accent); line-height:1.2; }}
code {{ background:#eef2ef; border-radius:4px; padding:.1rem .3rem; }}
.table-wrap {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; margin:1rem 0; }}
th,td {{ border:1px solid var(--line); padding:.55rem .65rem; text-align:left; }}
th {{ background:var(--panel); }}
img {{ max-width:48%; margin:.5rem 1% .5rem 0; border:1px solid var(--line);
border-radius:6px; }}
@media (max-width:760px) {{ .layout {{ display:block; }}
nav {{ position:static; margin-bottom:2rem; }} img {{ max-width:100%; }} }}
</style>
</head>
<body>
<header><h1>MMA3001: Short-Term Indoor CO2 Forecasting</h1>
<p>Ordered project documentation for engineering motivation, data, modelling,
validation, results, limitations and submission readiness.</p></header>
<div class="layout"><nav><strong>Contents</strong><ol>{navigation}</ol>
<p><a href="index.html">Code documentation</a></p></nav>
<main>{content}</main></div>
</body>
</html>'''

    report_path = output_dir / "project_report.html"
    report_path.write_text(document, encoding="utf-8")
    return report_path


def build_documentation():
    """Generate pdoc pages and the ordered project report in docs/."""
    import pdoc

    root = Path(__file__).resolve().parent
    output_dir = root / "docs"
    source = export_documentation_source(
        root / "CO2_Forecasting.ipynb",
        output_dir / "_source" / "co2_forecasting.py",
    )

    pdoc.render.configure(docformat="google", show_source=True, search=True)
    pdoc.pdoc(
        source,
        root / "co2_models.py",
        root / "build_docs.py",
        output_directory=output_dir,
    )
    report = build_project_report(root, output_dir)

    index_path = output_dir / "index.html"
    index_text = index_path.read_text(encoding="utf-8")
    index_text = index_text.replace(
        "<h2>Available Modules</h2>",
        '<h2>Project documentation</h2>'
        '<p><a href="project_report.html">Assessment report</a></p>'
        "<h2>Available Modules</h2>",
        1,
    )
    index_path.write_text(index_text, encoding="utf-8")
    return report


if __name__ == "__main__":
    print(build_documentation())
