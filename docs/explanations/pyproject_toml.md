# `pyproject.toml`: explained block by block

`pyproject.toml` is the standard file (PEP 518 / PEP 621) that describes a Python project: how to
build it, what it needs, and how the tools (pytest, ruff) are configured. Because of it,
`pip install -e .` works.

---

## Block 1: build system

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```

- **What:** tells `pip` which tool builds our package. Here it is `setuptools`.
- **Why setuptools:** it is the most common backend, and pip installs it automatically. `>=68` is
  a version that fully supports PEP 621 metadata and `src/` layouts **[from prior knowledge, to
  verify]**. In practice, `pip install -e .` worked here and on Python 3.10, 3.11 and 3.12. We do not pin it exactly
  because it is only a build tool, not a runtime library of the project.
- **Without it:** pip would fall back to an old legacy mode, and `pip install -e .` could fail or
  behave differently between machines **[from prior knowledge, to verify]**.

## Block 2: project metadata

```toml
[project]
name = "student_risk"
version = "0.1.0"
description = "..."
authors = [ { name = "Mohamed Reda Zarrouk" }, { name = "Georgy Salameh" } ]
requires-python = ">=3.10"
```

- `name`: the name pip uses for the package. It matches the import name `student_risk`.
- `version`: the project version, written in the format pip expects.
- `authors`: the two students.
- `requires-python = ">=3.10"`: pip refuses to install on Python 3.9 or older. We chose 3.10
  because every pinned library below still publishes wheels for 3.10, 3.11 and 3.12. We checked
  this on PyPI and by installing and running the tests on each version. Code features we use that
  need 3.10: `list[str]` and `tuple[...]` in type hints at runtime, and `X | Y` if we use it later.

## Block 3: runtime dependencies (pinned)

```toml
dependencies = [
    "numpy==2.2.6",
    "pandas==2.3.3",
    "scipy==1.15.3",
    "scikit-learn==1.7.2",
    "joblib==1.5.2",
    "PyYAML==6.0.3",
    "ucimlrepo==0.0.7",
    # Test and lint tools are regular dependencies on purpose: the graders run
    # "pip install -e ." then "pytest" in a fresh environment, with no extra.
    "pytest==9.2.0",
    "ruff==0.17.3",
]
```

- **What:** the libraries installed by `pip install -e .`.
- **Why `==` (pinned):** everybody (us and the teachers) gets exactly the same versions, so the
  results are the same and nothing breaks because a library changed.
- **Why these exact versions:** they are the newest releases that still support Python 3.10.
  Newer releases exist but dropped 3.10:
  - numpy 2.3+ needs Python ≥ 3.11 (2.5 needs ≥ 3.12)
  - pandas 3.x needs ≥ 3.11
  - scikit-learn 1.8+ needs ≥ 3.11
  - scipy 1.16+ needs ≥ 3.11
- What each library is for:
  - `numpy`: arrays and maths.
  - `pandas`: tables (DataFrames).
  - `scipy`: maths used inside scikit-learn.
  - `scikit-learn`: models, Pipeline, cross-validation.
  - `joblib`: saving and loading the trained model.
  - `PyYAML`: reading `config.yaml`.
  - `ucimlrepo`: downloading the dataset from UCI.
  - `pytest` (9.2.0, needs Python ≥ 3.10) and `ruff`: the test runner and the linter (see Block 4).
- **Without pins:** a teacher installing in a year could get pandas 4 and see different
  behaviour or errors.

## Block 4: why pytest and ruff are regular dependencies (Phase 3 change)

**Before Phase 3**, pytest and ruff were in an optional `[dev]` extra, installed with
`pip install -e ".[dev]"`.

**Why we moved them:** the subject says the teachers run, in a fresh environment,
`pip install -e .` and then `pytest`, and that a failure costs 2 points. With the extra, the
plain `pip install -e .` would not install pytest, so `pytest` would fail with "command not
found". Making them regular dependencies guarantees that the exact graded command sequence works.

**The cost:** a school that only wants predictions also installs a test runner and a linter
(a few MB). We accept that cost for a course project.

## Block 4b: the `student-risk` command (Phase 3)

```toml
[project.scripts]
student-risk = "student_risk.cli:main"
```

- **What:** when the package is installed, pip creates a small program called `student-risk`
  that calls the function `main` in `student_risk/cli.py`. It goes in `.venv/bin/` on Linux and
  macOS, and in `.venv\Scripts\` on Windows.
- **Why:** the README commands are short and the same on every system:
  `student-risk train`, `student-risk evaluate`, `student-risk predict --input ...`.
- **Without it:** you would type `python -m student_risk.cli train`. That still works and is
  given in the README as a fallback.
- **Checked:** the command works after `pip install -e .` in fresh environments on Python 3.10
  and 3.12.

## Block 5: where the package code is

```toml
[tool.setuptools.packages.find]
where = ["src"]
```

- **What:** tells setuptools to look for packages inside `src/`.
- **Why the `src/` layout:** the tests import the *installed* package, not a file that happens to
  be in the current folder. This catches packaging mistakes, such as a forgotten `__init__.py`.
- **Without it:** setuptools would look at the project root and not find `student_risk`.

## Block 6: pytest settings

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

- Running `pytest` with no arguments only searches the `tests/` folder. Without it, pytest would
  also search `.venv/` and be slow or collect the wrong files.

## Block 7: ruff (linter) settings

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "D"]
ignore = ["N803", "N806"]

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["D"]
```

- `line-length = 100`: the maximum line length. 100 is readable and avoids too many wrapped lines.
- `target-version = "py310"`: ruff only suggests syntax that works on our minimum Python.
- The rule groups in `select`:
  - `E`/`W` (pycodestyle): style errors and warnings.
  - `F` (pyflakes): real bugs such as unused imports or undefined names.
  - `I`: import sorting.
  - `N`: naming conventions.
  - `UP`: modern Python syntax.
  - `B` (bugbear): common bug patterns.
  - `D`: docstrings must exist on public functions and classes, which is a course requirement.
- `ignore = ["N803", "N806"]`: these two rules forbid capital letters in argument and variable
  names. We keep the scikit-learn convention `X` (a matrix) versus `y` (a vector), which every
  ML reader expects.
- `convention = "google"`: docstrings use the Google style, with `Args:` and `Returns:` sections.
- `"tests/*" = ["D"]`: test functions do not need docstrings, because their long names say what
  they check. The file `docs/explanations/test_*.md` explains each one.

## Block 8: formatter exclusion (Phase 2)

```toml
[tool.ruff.format]
exclude = ["*.md"]
```

- **What:** `ruff format` must not touch Markdown files.
- **Why:** ruff 0.17 also formats Python code blocks inside `.md` files (checked: it wanted to
  rewrite our docs). Our explanation snippets are copied from the source *with their original
  indentation*, for example a method shown indented inside its class. The formatter would remove
  that indentation, and the snippets would no longer match the code.
- Linting (`ruff check`) of `.py` files is not affected.

## Block 4c: the optional `notebook` extra (notebook phase)

```toml
[project.optional-dependencies]
notebook = [
    "jupyter==1.1.1",
    "nbconvert==7.17.2",
    "ipykernel==7.3.0",
    "matplotlib==3.10.9",
]
```

- **What:** extra tools needed only to run `notebooks/analysis.ipynb` and draw the figures.
  Install them with `pip install -e ".[notebook]"`.
- **Why optional:** the subject's graded sequence (`pip install -e .`, `pytest`, `train`,
  `evaluate`, `predict`) does not need Jupyter or matplotlib, which are large. Keeping them out
  keeps the main install light. We checked: a fresh main install has no matplotlib, and
  everything works.
- **What each one is for:**
  - `jupyter` is the meta-package (notebook interface);
  - `nbconvert` runs the notebook from the command line (`--execute`);
  - `ipykernel` is the Python kernel that executes the cells;
  - `matplotlib` draws the figures (used only by `plots.py`).
- **Pinned to the newest versions that still support Python 3.10 (checked on PyPI):**
  - matplotlib 3.11 and ipykernel 7.4 need Python ≥ 3.11, so we use 3.10.9 and 7.3.0;
  - `ipykernel` is pinned explicitly, otherwise Python 3.12 would get 7.4 and Python 3.10 would
    get 7.3.
- **Checked:** the extra installs and the notebook executes in fresh environments on Python 3.10
  and 3.12.
