---
description: Update all documentation files with recent changes
allowed-tools: Read, Edit, Write, Glob, Grep
---

Please update all the docs (.mdx and .md) with all the changes.

**Files to update:**

1. **Root documentation:**
   - `CLAUDE.md` - Claude Code guidance
   - `README.md` - Main project README
   - `COMPANY_DB.md` - The companies database

2. **Python library documentation:**
   - `statement-extractor-lib/README.md` - Library README
   - `statement-extractor-lib/CLAUDE.md` - Claude Code Guidance

3. **Runpod Documentation:**
   - `runpod/README.md`

4. **Local Server Documentation:**
   - `local-server/README.md`
   
4. **Website documentation (MDX):**
   - `src/app/docs/sections/*.mdx`
   - `src/components/documentation.tsx`
   - `src/components/llm-prompts.tsx`
   - `src/components/pipeline-diagram.tsx`
**Process:**

$ARGUMENTS$

1. First, review recent code changes to understand what needs documenting.
2. Search for any inconsistencies between code and documentation
3. Update all relevant documentation files
4. Ensure version numbers, feature descriptions, and examples are accurate
5. Verify pipeline stages, plugins, and API signatures match the code
