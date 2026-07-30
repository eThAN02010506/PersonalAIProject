# Security Policy

## Supported version

Qwopus-Agent is currently pre-1.0. Security fixes are applied to the latest `master` revision.

## Reporting a vulnerability

Do not open a public issue for vulnerabilities involving authentication, authorization, local file
access, browser network boundaries, pandas sandbox escape, prompt or Tool Observation exposure,
credentials, or cross-account knowledge leakage.

Use the repository's
[private vulnerability reporting](https://github.com/eThAN02010506/PersonalAIProject/security/advisories/new)
instead. Include:

- the affected revision and environment;
- the smallest reproducible input;
- expected and observed authorization boundary;
- impact and whether sensitive data was exposed;
- logs with credentials, document bodies, and personal data removed.

The maintainer will acknowledge a complete report when it is reviewed, validate the impact, and
coordinate a fix before public disclosure. Do not test against systems or accounts you do not own.

## Deployment boundary

Qwopus-Agent is local-first, not automatically offline. A remote model receives prompts sent to
that endpoint, Tavily receives authorized search queries, and first-time model setup may download
artifacts. The Debug Console contains sensitive traces and is intentionally restricted to a
loopback client and an administrator account.
