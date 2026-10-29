# `src/student_risk/__init__.py`: explained line by line

```python
"""student_risk: early-warning classifier for students at risk of failing."""
```

- **What:** the docstring of the package. It appears in `help(student_risk)`.
- **Why:** ruff's docstring rules (`D`) require a docstring for every public module, including the
  package itself.

```python
__version__ = "0.1.0"
```

- **What:** the version of the package, readable with `student_risk.__version__`.
- **Why:** a common Python convention. Keep it equal to `version` in `pyproject.toml`.

**Why does the file exist at all?** A folder with an `__init__.py` is a *regular package*.
setuptools' `packages.find` only finds folders that contain one. Without it, `pip install -e .`
would install nothing, and `import student_risk` would fail.
