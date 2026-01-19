---
description: Run all tests for the project
allowed-tools: Bash(uv:*), Bash(pnpm:*), Bash(pytest:*)
---

Run tests for the statement-extractor project:

**Python Library Tests:**
```bash
cd statement-extractor-lib && uv run pytest -v
```

**Frontend Tests (if applicable):**
```bash
pnpm lint
pnpm build
```

**Test with specific file or pattern:**
If $ARGUMENTS is provided, run tests matching that pattern:
```bash
cd statement-extractor-lib && uv run pytest -v -k "$ARGUMENTS"
```

After running tests:
1. Report test results (passed/failed/skipped)
2. If failures, analyze the error messages
3. Suggest fixes for any failing tests
4. Check test coverage if available