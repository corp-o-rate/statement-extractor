---
description: Publish the Python library to PyPI
allowed-tools: Bash(uv:*), Bash(git:*), Read, Edit
---

Publish the corp-extractor library to PyPI:

**Pre-publish Checklist:**

1. **Version Check:**
   - Read `statement-extractor-lib/pyproject.toml`
   - Verify version number is incremented appropriately do this by checking PyPi - https://pypi.org/project/corp-extractor/ - use semantic versioning bump the middle number for non-minor functional changes and the final number for non-functional changes (fixes, tidy up etc.) 

Make sure `statement-extractor-lib/src/statement_extractor/__init__.py` has the correct same version number.

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

5. **Check Documentation:**

   Make sure these files are accurate and up to date.
   - `statement-extractor-lib/README.md` - Library README
   - `statement-extractor-lib/CLAUDE.md` - Claude Code Guidance
   
   Update the main documentation with the new library version number
   update the `runpod/Dockerfile` with the new version number

   Update the files in `notebooks` to make sure the documentation and code in the notebooks match the release.

6. **Publish:**
   ```bash
   cd statement-extractor-lib && uv publish
   ```
   
   Then commit and push to Github. 

7. **Post-publish:**
   - Tag the release in git: `git tag v{version}`
   - Push tags: `git push --tags`

**Note:** Requires PyPI credentials configured in environment.
