# Contributing to Surforama
## developer installation

If you would like to make changes to the surforama source code, you can install surformama with the developer tools as follows:

```bash
cd /path/to/your/surforama/source/code/folder
pip install -e ".[dev]"
```
We use pre-commit to keep the code tidy. Install the pre-commit hooks to activate the checks:

```bash
pre-commit install
```
