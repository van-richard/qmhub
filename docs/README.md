# Compiling QMHub's Documentation

The docs for this project are built with [Sphinx](https://www.sphinx-doc.org/).
Install the documentation dependencies from the repository root:

```bash
python -m pip install -r docs/requirements.txt
```

Then build the static HTML pages from the `docs` directory:

```bash
cd docs
make html
```

The generated HTML output is written to `docs/_build/html/`. Open
`docs/_build/html/index.html` in a browser to view the local documentation.
