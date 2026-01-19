---
description: Publish the Python library to PyPI
allowed-tools: Bash(uv:*), Bash(git:*), Read, Edit
---

Publish the corp-extractor library to PyPI:

**Pre-publish Checklist:**

1. **Version Check:**
   - Read `statement-extractor-lib/pyproject.toml`
   - Verify version number is incremented appropriately

2. **Run Tests:**
   ```bash
   cd statement-extractor-lib && uv run pytest
   ```

3. **Build Package:**
   ```bash
   cd statement-extractor-lib && uv build
   ```

4. **Verify Build:**
   - Check dist/ directory for wheel and tarball
   - Verify package contents look correct

5. **Publish:**
   ```bash
   cd statement-extractor-lib && uv publish
   ```

6. **Post-publish:**
   - Tag the release in git: `git tag v{version}`
   - Push tags: `git push --tags`
   - Update documentation if needed

**Note:** Requires PyPI credentials configured in environment.
