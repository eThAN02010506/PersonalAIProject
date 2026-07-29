# Qwopus-Agent Development Rules

You are working on Qwopus-Agent, a local-first modular AI Agent framework.

Follow these principles:

- Plan before acting.
- Use tools only through registered tool interfaces.
- Keep model behavior independent from the concrete model backend.
- Prefer small, inspectable steps over hidden complexity.
- Record artifacts and decisions through the existing report and storage boundaries when required.
## 1. THINK

- **Pause**: State understanding & strategy in 2 sentences before writing code.
- **Radius**: Trace upstream/downstream impact. Ask if dependencies are unclear.
- **KISS**: Propose the solution with the fewest lines. Zero overengineering.

## 2. EDIT
- **Strict**: Write ONLY what was asked. No future-proofing or abstractions.
- **Surgical**: Modify target lines only. Never touch adjacent working code.
- **Purge**: Delete any imports, variables, or functions made dead by your change.

## 3. VERIFY
- **Check**: State a 1-line verification plan before touching code.
- **Prove**: Verify via tests OR trace the logic flow step-by-step to guarantee zero regressions.

## 4. COMMENT

- Explain why non-obvious code is needed and what role it serves.
- Do not add comments that merely restate the code.
