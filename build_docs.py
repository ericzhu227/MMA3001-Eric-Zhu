"""Generate pdoc HTML documentation from the current notebook source.

Run ``python build_docs.py`` in the environment containing pdoc (for example,
the Personal conda environment). The notebook's data-processing cells are never
executed during documentation generation.
"""
from pathlib import Path
import ast
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
        'Training/validation/test splits are chronological; final test scoring '
        'and trained model comparisons remain future work.',
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
    pdoc.pdoc(source, root / 'build_docs.py', output_directory=output_dir)
    return output_dir / 'index.html'


if __name__ == '__main__':
    print(build_documentation())
