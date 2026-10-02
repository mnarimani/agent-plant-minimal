"""AgentPlant backend core.

Clean, dependency-light building blocks for the Streamlit app in ``app.py``:

- ``agent``          : the continue / draft / complete conversational agent
                        that produces ``dynamics(t, x, u)`` Python code.
- ``json_extract``    : permissive JSON extraction from raw LLM text.
- ``openai_client``   : thin, logged wrapper around the OpenAI Responses API.
- ``rag``             : OpenAI-native RAG (Files + Vector Store + file_search).
- ``websearch``       : OpenAI-native web_search with a focused-query gate.
- ``sandbox``         : restricted exec of drafted dynamics() + ODE simulation.
- ``steps``           : small dataclass used to render the staged step log.
- ``logging_utils``   : prompt/response logging under ``.logs/``.
"""

__all__ = [
    "agent",
    "json_extract",
    "openai_client",
    "rag",
    "websearch",
    "sandbox",
    "steps",
    "logging_utils",
]
