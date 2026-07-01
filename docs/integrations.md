# Client Integrations Guide

Herd is designed to be fully OpenAI-compatible. This means you can drop it in as a local replacement for OpenAI or Ollama in almost any modern application, SDK, or UI framework.

By default, Herd listens on:
*   **API Base URL**: `http://127.0.0.1:11434`
*   **OpenAI Base URL**: `http://127.0.0.1:11434/v1`

---

## 1. UIs & Desktop Clients

### Open WebUI
[Open WebUI](https://github.com/open-webui/open-webui) is a popular, feature-rich web user interface for local LLMs.
To connect Open WebUI to Herd:
1. Open your Open WebUI dashboard.
2. Navigate to **Settings** ➔ **Connections**.
3. Under the **OpenAI API** section:
    *   Set the **API URL** to `http://host.docker.internal:11434/v1` (if running Open WebUI in Docker) or `http://127.0.0.1:11434/v1` (if running locally).
    *   Set the **API Key** to any arbitrary string (e.g. `herd`).
4. Click **Save** (or the refresh button).
5. The models downloaded in Herd (visible in `herd list`) will now appear in your model selection dropdown!

### Chatbox / NextChat / LibreChat
For popular desktop chat interfaces:
1. Go to settings and choose **OpenAI** as the provider (or API Type).
2. Set the **API Host** (or Base URL) to `http://127.0.0.1:11434`. (Ensure you append `/v1` if the client requires the full path).
3. Set any placeholder text as the API Key.

---

## 2. Python SDKs

### OpenAI Python SDK
You can use the official `openai` Python library directly with Herd by changing the `base_url`.

```python
from openai import OpenAI

# Initialize client pointing to Herd gateway
client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="herd-local-token"  # Any string works
)

# Chat completions call (automatically loads the model if idle)
response = client.chat.completions.create(
    model="unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M",
    messages=[
        {"role": "user", "content": "Write a 3-word poem."}
    ],
    stream=False
)

print(response.choices[0].message.content)
```

### LangChain
Connect Herd to LangChain pipelines using the `ChatOpenAI` wrapper.

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M",
    openai_api_base="http://127.0.0.1:11434/v1",
    openai_api_key="local-placeholder"
)

response = llm.invoke("What is the speed of light?")
print(response.content)
```

### LlamaIndex
Integrate Herd with LlamaIndex for local Retrieval-Augmented Generation (RAG).

```python
from llama_index.llms.openai import OpenAI

llm = OpenAI(
    model="unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M",
    api_base="http://127.0.0.1:11434/v1",
    api_key="local-placeholder"
)

response = llm.complete("Explain RAG in one sentence.")
print(response)
```

---

## 3. Developer & IDE Extensions

### Continue (VS Code & JetBrains)
[Continue](https://github.com/continuedev/continue) is an open-source autopilot extension for IDEs.
To use Herd models for code completions and chat:
1. Open your `~/.continue/config.json` configuration file.
2. Add your model to the `models` list:

```json
{
  "models": [
    {
      "title": "Herd Qwen",
      "provider": "openai",
      "model": "unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M",
      "apiBase": "http://127.0.0.1:11434/v1"
    }
  ],
  "tabAutocompleteModel": {
    "title": "Herd Autocomplete",
    "provider": "openai",
    "model": "unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M",
    "apiBase": "http://127.0.0.1:11434/v1"
  }
}
```
