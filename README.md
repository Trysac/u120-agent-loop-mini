# agent-loop-mini

[![PyPI version](https://img.shields.io/pypi/v/trysac-agent-loop-mini)](https://pypi.org/project/trysac-agent-loop-mini/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-agent-loop-mini)](LICENSE)
[![CI](https://github.com/Trysac/u120-agent-loop-mini/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u120-agent-loop-mini/actions/workflows/ci.yml)

Minimal tool-using agent loop over a text-only LLM port: JSON actions, step, time and context budgets, loop detection, escaped tool results and content-free events

## Problem

Describe in two or three sentences the concrete problem this utility solves, who hits it, and what
they do today without it. Replace this paragraph when the first real feature lands.

## Install

```sh
pip install trysac-agent-loop-mini
# or
uv add trysac-agent-loop-mini
```

Requires Python 3.10 or newer. No runtime dependencies.

## Usage

```python
import agent_loop_mini

# normalize() trims surrounding whitespace and lower-cases the text.
print(agent_loop_mini.normalize("  Hello World "))  # -> "hello world"

# Blank input is rejected explicitly instead of returning an empty string.
try:
    agent_loop_mini.normalize("   ")
except ValueError as exc:
    print(f"rejected: {exc}")
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [decisions](docs/decisions/)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md)
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez
