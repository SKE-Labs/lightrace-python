<p align="center">
  <img src="https://raw.githubusercontent.com/SKE-Labs/lightrace/main/packages/frontend/public/white_transparent.png" alt="LightRace" width="280" />
</p>

<h1 align="center">lightrace-python</h1>

<p align="center">
  <a href="https://pypi.org/project/lightrace/"><img src="https://img.shields.io/pypi/v/lightrace?style=flat-square&color=ff1a1a" alt="PyPI version" /></a>
  <a href="https://github.com/SKE-Labs/lightrace-python/stargazers"><img src="https://img.shields.io/github/stars/SKE-Labs/lightrace-python?style=flat-square" alt="GitHub stars" /></a>
  <a href="https://github.com/SKE-Labs/lightrace-python/blob/main/LICENSE"><img src="https://img.shields.io/github/license/SKE-Labs/lightrace-python?style=flat-square" alt="License" /></a>
</p>

<p align="center">Lightweight LLM tracing SDK for Python with remote tool invocation.</p>

---

## Install

```bash
pip install lightrace
```

## Quick Start

```python
from lightrace import Lightrace, trace

lt = Lightrace(
    public_key="pk-lt-demo",
    secret_key="sk-lt-demo",
    host="http://localhost:3000",
)

# Root trace
@trace()
def run_agent(query: str):
    return search(query)

# Span
@trace(type="span")
def search(query: str) -> list:
    return ["result1", "result2"]

# Generation (LLM call)
@trace(type="generation", model="gpt-4o")
def generate(prompt: str) -> str:
    return "LLM response"

# Tool — remotely invocable from the Lightrace UI
@trace(type="tool")
def weather_lookup(city: str) -> dict:
    return {"temp": 72, "unit": "F"}

# Tool — traced but NOT remotely invocable
@trace(type="tool", invoke=False)
def read_file(path: str) -> str:
    return open(path).read()

run_agent("hello")
lt.flush()
lt.shutdown()
```

## `@trace` API

```python
@trace()                                    # Root trace
@trace(type="span")                         # Span observation
@trace(type="generation", model="gpt-4o")   # LLM generation
@trace(type="tool")                         # Tool (remotely invocable)
@trace(type="tool", invoke=False)           # Tool (trace only)
```

### Parameters

| Parameter  | Type   | Default | Description                                              |
| ---------- | ------ | ------- | -------------------------------------------------------- |
| `type`     | `str`  | `None`  | `"span"`, `"generation"`, `"tool"`, `"chain"`, `"event"` |
| `name`     | `str`  | `None`  | Override name (defaults to function name)                 |
| `invoke`   | `bool` | `True`  | For `type="tool"`: register for remote invocation        |
| `model`    | `str`  | `None`  | For `type="generation"`: LLM model name                  |
| `metadata` | `dict` | `None`  | Static metadata attached to every call                   |

## Compatibility

Lightrace server also accepts traces from Langfuse Python/JS SDKs.

## Related

- [Lightrace](https://github.com/SKE-Labs/lightrace) — the main platform (backend + frontend)
- [Lightrace CLI](https://github.com/SKE-Labs/lightrace-cli) — self-host with a single command
- [lightrace-js](https://github.com/SKE-Labs/lightrace-js) — TypeScript/JavaScript SDK

## Development

```bash
uv sync --extra dev
uv run pre-commit install
uv run pytest -s -v tests/
uv run ruff check .
uv run mypy src/lightrace
```

## License

MIT
