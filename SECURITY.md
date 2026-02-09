# Security Policy

Thanks for helping keep AbstractAgent users safe.

AbstractAgent is part of the **AbstractFramework** ecosystem (overview): https://github.com/lpalbou/AbstractFramework

## Reporting a vulnerability

Please report suspected security vulnerabilities **privately** and give maintainers time to investigate and issue a fix.

Preferred reporting channels:

1) **Private security advisory** (recommended when available): use your repository hosting provider’s private security reporting feature.
2) **Email**: if a private advisory is not available, email `contact@abstractcore.ai` with the details and request security routing.

## What to include

To help us triage quickly, include:

- A clear description of the issue and potential impact
- Steps to reproduce (proof-of-concept if possible)
- Affected versions (see `pyproject.toml` and `CHANGELOG.md`)
- Environment details (OS, Python version, provider/model if relevant)
- Any suggested fix or mitigation

## Coordinated disclosure

Please do not open public issues or disclose the vulnerability publicly until a fix is released (or a coordinated disclosure date is agreed).

## Security notes for users (tool execution)

Some default tools can execute code or shell commands, depending on your host/runtime policy:
- `execute_python`: local subprocess with a timeout (`src/abstractagent/sandbox/local.py`) — not a hardened sandbox.
- `execute_command`: runs shell commands (policy is controlled by your tool executor / host allowlist).

See also: [`docs/tools.md`](docs/tools.md) and [`docs/getting-started.md`](docs/getting-started.md).
