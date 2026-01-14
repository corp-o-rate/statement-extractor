# Publishing to PyPI

This guide explains how to publish the `statement-extractor` package to PyPI.

## Prerequisites

1. **PyPI Account**: Create an account at https://pypi.org/account/register/
2. **API Token**: Generate a token at https://pypi.org/manage/account/token/
   - Scope: "Entire account" (for first publish) or project-specific after first publish
   - Save this token securely - you'll only see it once!

## First-Time Setup

### Option 1: Using uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set your PyPI token (add to ~/.zshrc or ~/.bashrc for persistence)
export UV_PUBLISH_TOKEN=pypi-xxxxxxxxxxxxx
```

### Option 2: Using Traditional Tools

```bash
# Install build tools
pip install build twine

# Configure PyPI credentials in ~/.pypirc
cat > ~/.pypirc << EOF
[pypi]
username = __token__
password = pypi-xxxxxxxxxxxxx
EOF

chmod 600 ~/.pypirc
```

## Publishing Steps

### 1. Update Version

Edit `pyproject.toml` and increment the version:
```toml
version = "0.1.1"  # Increment for each release
```

Follow [semantic versioning](https://semver.org/):
- MAJOR (1.0.0): Breaking changes
- MINOR (0.2.0): New features, backwards compatible
- PATCH (0.1.1): Bug fixes, backwards compatible

### 2. Build the Package

```bash
cd statement-extractor-lib

# Using uv
uv build

# Or using traditional tools
python -m build
```

This creates files in `dist/`:
- `statement_extractor-0.1.0.tar.gz` (source distribution)
- `statement_extractor-0.1.0-py3-none-any.whl` (wheel)

### 3. Test on TestPyPI (Optional but Recommended)

```bash
# Using uv
uv publish --publish-url https://test.pypi.org/legacy/

# Or using twine
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ statement-extractor
```

### 4. Publish to PyPI

```bash
# Using uv
uv publish

# Or using twine
twine upload dist/*
```

### 5. Verify

```bash
# Wait a few minutes, then:
pip install statement-extractor

# Test it works
python -c "from statement_extractor import extract_statements; print('Success!')"
```

## Automated Publishing (GitHub Actions)

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Build package
        run: |
          cd statement-extractor-lib
          uv build

      - name: Publish to PyPI
        env:
          UV_PUBLISH_TOKEN: ${{ secrets.PYPI_API_TOKEN }}
        run: |
          cd statement-extractor-lib
          uv publish
```

Then add `PYPI_API_TOKEN` to your repository secrets:
1. Go to GitHub repo → Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Name: `PYPI_API_TOKEN`
4. Value: Your PyPI API token (starting with `pypi-`)

## Troubleshooting

### "Package name already exists"
The name `statement-extractor` might already be taken on PyPI. You may need to use a different name like:
- `statement-extractor-nlp`
- `corp-o-rate-statement-extractor`
- `t5-statement-extractor`

Update the `name` field in `pyproject.toml` accordingly.

### "Invalid or non-existent authentication"
- Ensure your token is correct and hasn't expired
- For uv: Check `UV_PUBLISH_TOKEN` is set
- For twine: Check `~/.pypirc` has correct credentials

### "Version already exists"
You can't upload the same version twice. Increment the version number in `pyproject.toml`.

### Build fails
```bash
# Clean build artifacts
rm -rf dist/ build/ *.egg-info/

# Try again
uv build
```

## Quick Reference

```bash
# Full publish workflow
cd statement-extractor-lib
uv build
uv publish

# With version bump
sed -i '' 's/version = "0.1.0"/version = "0.1.1"/' pyproject.toml
uv build
uv publish
```

## Links

- PyPI: https://pypi.org/
- TestPyPI: https://test.pypi.org/
- uv docs: https://docs.astral.sh/uv/
- Python Packaging Guide: https://packaging.python.org/