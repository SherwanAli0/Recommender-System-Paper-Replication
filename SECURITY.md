# Security Policy

This is a research-replication repository, not a production system. There is no exposed network service, no authentication layer, and no sensitive data committed. The risk surface is low.

That said, if you discover a security-relevant issue (for example: a path-traversal bug in one of the data loaders, or a malicious-file-handling weakness in the dataset preprocessing), please report it responsibly.

## How to report

1. **Do NOT open a public issue** that describes the vulnerability.
2. Instead, open a private email to one of the authors listed in `CITATION.cff`.
3. Include:
   - A description of the issue
   - The minimal reproducing input
   - The version (commit hash) where you observed the issue
4. We will acknowledge within 7 days and aim to release a fix within 30 days for verified issues.

## Out of scope

The following are not considered security issues for this project:

- Numerical differences between our results and the paper (these are documented in `Person_2_Faithful_FUS/REPRO_DEBUG_NOTES.md`)
- Performance issues (slow runs)
- Compatibility issues with very old or very new Python versions outside the supported range
- Use of permissive licenses (MIT / CC-BY 4.0) that allow commercial reuse
