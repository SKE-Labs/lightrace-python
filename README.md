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

## Integrations

### OpenAI

```python
import openai
from lightrace import Lightrace, trace
from lightrace.integrations.openai import LightraceOpenAIInstrumentor

lt = Lightrace(
    public_key="pk-lt-demo",
    secret_key="sk-lt-demo",
    host="http://localhost:3000",
)

client = openai.OpenAI()
instrumentor = LightraceOpenAIInstrumentor(client=lt)
instrumentor.instrument(client)

@trace()
def ask_gpt():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=256,
        messages=[{"role": "user", "content": "What is the speed of light?"}],
    )
    return response.choices[0].message.content

ask_gpt()
lt.flush()
lt.shutdown()
```

### Anthropic

```python
import anthropic
from lightrace import Lightrace, trace
from lightrace.integrations.anthropic import LightraceAnthropicInstrumentor

lt = Lightrace(
    public_key="pk-lt-demo",
    secret_key="sk-lt-demo",
    host="http://localhost:3000",
)

client = anthropic.Anthropic()
instrumentor = LightraceAnthropicInstrumentor(client=lt)
instrumentor.instrument(client)

@trace()
def ask_claude():
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{"role": "user", "content": "What is the capital of Mongolia?"}],
    )
    return response.content[0].text

ask_claude()
lt.flush()
lt.shutdown()
```

### LangChain

```python
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from lightrace import Lightrace
from lightrace.integrations.langchain import LightraceCallbackHandler

lt = Lightrace(
    public_key="pk-lt-demo",
    secret_key="sk-lt-demo",
    host="http://localhost:3000",
)

handler = LightraceCallbackHandler(client=lt)
model = ChatOpenAI(model="gpt-4o-mini", max_tokens=256)

response = model.invoke(
    [HumanMessage(content="What is the speed of light?")],
    config={"callbacks": [handler]},
)

lt.flush()
lt.shutdown()
```

### Claude Agent SDK

```python
import anyio
from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock
from lightrace import Lightrace
from lightrace.integrations.claude_agent_sdk import traced_query

lt = Lightrace(
    public_key="pk-lt-demo",
    secret_key="sk-lt-demo",
    host="http://localhost:3000",
)

async def main():
    async for message in traced_query(
        prompt="What files are in the current directory?",
        options=ClaudeAgentOptions(max_turns=3),
        client=lt,
        trace_name="file-lister",
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text)
        elif isinstance(message, ResultMessage):
            print(f"Cost: ${message.total_cost_usd:.4f}")

    lt.flush()
    lt.shutdown()

anyio.run(main)
```

You can also use the handler directly for more control:

```python
from claude_agent_sdk import query
from lightrace.integrations.claude_agent_sdk import LightraceAgentHandler

handler = LightraceAgentHandler(prompt="Hello", client=lt, trace_name="my-agent")

async for message in query(prompt="Hello"):
    handler.handle(message)
```

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
