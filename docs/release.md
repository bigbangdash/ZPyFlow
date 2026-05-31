# Release Process

ZPyFlow publishes CPython 3.10+ wheels plus an sdist through maturin.

## Package Metadata

- Package name: `zpyflow`
- Python support: CPython 3.10+
- License: MIT
- Optional extras:
  - `zpyflow[numpy]` installs NumPy support
  - `zpyflow[arrow]` installs PyArrow support
  - `zpyflow[dev]` installs test, benchmark, type-check, and release tooling

## Local Checks

Run before tagging:

```bash
docker compose run --rm test sh -c \
  'maturin develop --release && pytest tests/ -q && mypy zpyflow tests/typing_examples.py && pyright tests/typing_examples.py'
```

Build local artifacts:

```bash
maturin build --release --sdist
```

Note: abi3 wheels are a release goal, but they are blocked until the numpy
buffer constructors and lazy list extraction avoid APIs unavailable under
`Py_LIMITED_API`. Current releases should use CPython-version-specific wheels.

Test a wheel in a clean environment:

```bash
python -m venv /tmp/zpyflow-install-check
/tmp/zpyflow-install-check/bin/python -m pip install --upgrade pip
/tmp/zpyflow-install-check/bin/python -m pip install dist/zpyflow-*.whl
/tmp/zpyflow-install-check/bin/python -c "from zpyflow import Query, col; assert Query([1, 2, 3]).filter(col > 1).to_list() == [2, 3]"
```

## TestPyPI

Use trusted publishing from the `release.yml` workflow when possible. For a
manual dry run:

```bash
maturin publish --repository testpypi --skip-existing
```

Install-check from TestPyPI:

```bash
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ zpyflow
```

## PyPI

After TestPyPI install checks pass, create a GitHub Release for a tag like
`v0.1.0`. The release workflow builds Linux, macOS, and Windows wheels and
publishes them to PyPI.
