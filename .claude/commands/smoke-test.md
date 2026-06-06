Run a post-release smoke test for ZPyFlow. The argument is the version to test (e.g. `0.1.2`).

## Steps

**1. Create a fresh venv**
```bash
python3 -m venv /tmp/zpyflow_smoke_$VERSION
```

**2. Install the released package**
```bash
/tmp/zpyflow_smoke_$VERSION/bin/pip install --quiet zpyflow==$VERSION
```
If pip cannot find the version, it may not be on PyPI yet. Wait 2–5 minutes and retry.

**3. Run smoke tests**

Execute the following inline script via the venv Python:

```bash
/tmp/zpyflow_smoke_$VERSION/bin/python3 - <<'EOF'
import zpyflow
from zpyflow import Query, col, field

# version check
assert zpyflow.__version__ == "$VERSION", f"got {zpyflow.__version__}"

# numeric fast path
result = Query([1.0, 2.0, 3.0, 4.0, 5.0]).filter(col > 2).to_list()
assert result == [3.0, 4.0, 5.0], f"numeric filter failed: {result}"

# sum / mean
assert Query([1.0, 2.0, 3.0]).sum() == 6.0
assert Query([1.0, 2.0, 3.0]).mean() == 2.0

# dict / field DSL
rows = [{"x": i} for i in range(5)]
result = Query(rows).filter(field("x") > 2).to_list()
assert len(result) == 2

# spec-077 methods
result = Query([1, None, 2, None, 3]).compact().to_list()
assert result == [1, 2, 3], f"compact failed: {result}"

result = Query([3.0, 1.0, 4.0, 1.0, 5.0]).median()
assert result == 3.0, f"median failed: {result}"

result = Query([1, 2, 3, 4]).product()
assert result == 24, f"product failed: {result}"

print(f"All smoke tests passed for zpyflow {zpyflow.__version__}")
EOF
```

**4. Delete the venv**
```bash
rm -rf /tmp/zpyflow_smoke_$VERSION
```

**5. Report result**
- If all assertions passed: print "smoke test passed for v$VERSION"
- If any assertion failed: print the error, remind the user the venv has been deleted, and suggest opening a bug issue.
