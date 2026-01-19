---
description: Run the extraction pipeline on text
allowed-tools: Bash(uv:*)
---

Run the statement extraction pipeline on the provided text:

```bash
cd statement-extractor-lib && uv run corp-extractor pipeline "$ARGUMENTS" --json
```

If $ARGUMENTS is a file path, use:
```bash
cd statement-extractor-lib && uv run corp-extractor pipeline -f "$ARGUMENTS" --json
```

After running:
1. Display the extracted statements
2. Show entity types and confidence scores
3. Report any taxonomy classifications
4. Highlight any errors or warnings
