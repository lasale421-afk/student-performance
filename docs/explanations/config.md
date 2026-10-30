# `src/student_risk/config.py`: explained line by line

**Role:** read `config.yaml` once and give the rest of the code a typed object (`Config`) instead of
raw dictionaries and hard-coded values.

---

## Imports

```python
"""Load project settings from config.yaml into a typed Config object."""
```

The module docstring, required by ruff (`D100`), says in one line what the file is for.

```python
from dataclasses import dataclass
from pathlib import Path

import yaml
```

- `dataclass`: generates `__init__`, `__repr__` and `__eq__` for a class that only stores data.
  Without it, we would have to write a long `__init__` by hand.
- `Path`: object-oriented file paths that work on Linux, macOS and Windows, because they use the
  right separator.
- `yaml` (from PyYAML): parses the YAML file into Python dicts and lists.
- The blank line between the standard-library imports and `yaml` is the import order that ruff
  (`I`) enforces: standard library first, then third-party, then our own package.

## Constants

```python
# src/student_risk/config.py -> parents[2] is the project root folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
```

- `__file__` is the path of this file. `.resolve()` makes it absolute.
- `.parents[0]` is `src/student_risk/`, `.parents[1]` is `src/`, and `.parents[2]` is the
  project root.
- **Why:** this is how we avoid hard-coded paths such as `/home/reda/...`. The project finds its
  own config wherever it is cloned, on any machine.
- **Limit (honest):** this works with the editable install (`pip install -e .`) we document,
  because the code stays in `src/`. With a non-editable install, the file would live in
  `site-packages`, and you would pass the config path explicitly.

## The `Config` dataclass

```python
@dataclass(frozen=True)
class Config:
    """All project settings, read once from config.yaml. ..."""

    random_seed: int
    uci_id: int
    cache_path: Path
    grade_column: str
    pass_mark: int
    leakage_columns: list[str]
    test_size: float
    cv_splits: int
    target_recall: float
    tradeoff_recalls: list[float]
    reports_dir: Path
    model_path: Path
    grids: dict[str, dict[str, list]]
```

- `@dataclass` turns the annotated class attributes into constructor arguments:
  `Config(random_seed=42, ...)`.
- `frozen=True` makes the object read-only after creation: `config.random_seed = 7` raises an
  error. **Why:** no part of the code can secretly change a setting halfway through a run.
- The type hints (`int`, `Path`, `list[str]`...) document each field and let an editor or type
  checker catch mistakes, for example passing a string where a number is expected.
- The docstring lists every attribute, because ruff requires docstrings on public classes and
  the teachers asked for them.
- **Why a class and not a plain dict:**
  - `config.pass_mark` is checked by the editor, but `raw["data"]["pass_mark"]` is not.
  - A typo in a field name fails immediately, instead of deep inside a function.
- **Why the tests like it:** they can build a `Config(...)` by hand with a temporary cache path,
  with no YAML file at all (see `conftest.md`).

## `load_config`

```python
def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
```

The function takes an optional path, so tests or a teacher can use another file. By default it
reads the project's `config.yaml`. It returns a `Config`.

```python
    with open(path, encoding="utf-8") as file:
        raw = yaml.safe_load(file)
```

- `with open(...)` closes the file automatically, even if an error happens.
- `encoding="utf-8"` gives the same result on Windows, whose default encoding differs **[from
  prior knowledge, to verify]**.
- `yaml.safe_load` only builds plain Python types (dict, list, str, int...). **Why not
  `yaml.load`:** it can create arbitrary Python objects, which is a security risk if the file is
  edited by someone else **[from prior knowledge, to verify]**.

```python
    root = Path(path).resolve().parent
```

This is the folder that contains the config file. The relative paths written in the YAML (such as
`data/student_performance.csv`) are taken relative to this folder. So the result does not depend
on the folder you run the command from.

```python
    return Config(
        random_seed=raw["random_seed"],
        uci_id=raw["data"]["uci_id"],
        cache_path=root / raw["data"]["cache_path"],
        ...
        target_recall=raw["threshold"]["target_recall"],
        tradeoff_recalls=list(raw["threshold"]["tradeoff_recalls"]),
        reports_dir=root / raw["paths"]["reports_dir"],
        model_path=root / raw["paths"]["model_path"],
        grids=raw["grids"],
    )
```

- Each YAML key is copied explicitly into a field. The YAML is grouped in sections (`data`,
  `split`, `cv`, `threshold`) to be readable for humans, while `Config` is flat to be easy to use
  in code.
- `root / ...` builds an absolute path with the `/` operator of `Path`.
- `list(...)` around `leakage_columns` guarantees a real Python list.
- **If a key is missing** in the YAML, you get a `KeyError` with the key's name right at startup.
  This is the behaviour we want: fail early and clearly.

## Phase 2 additions

- `reports_dir: Path`: the folder for result CSVs. Like `cache_path`, it is built as
  `root / ...`, so it is absolute.
- `grids: dict[str, dict[str, list]]`: the type hint reads "a dict from a model name to (a dict
  from a parameter name to a list of values)". For example:

  ```python
  config.grids["knn"] == {"clf__n_neighbors": [5, 11, 21, 31], "clf__weights": ["uniform", "distance"]}
  ```

  It is copied as is from YAML (`raw["grids"]`), because it already has exactly the shape
  `GridSearchCV(param_grid=...)` expects.
- `tradeoff_recalls: list[float]` (Phase 2b): the recall targets compared in
  `reports/threshold_tradeoff.csv`. `list(...)` guarantees a real list, as for
  `leakage_columns`.
- `model_path: Path` (Phase 3): the absolute path of the saved model file. It is built with
  `root / ...` like the other paths, so the tests can point it into a temporary folder.
