# Prompts

Every text the app sends to an LLM is in this folder. The code holds none; it loads them with
`backend_core.prompts` (`text(file, key)`, `fill(template, name=value)`).

```
prompts/
├── chat/                         the plant-model chat agent
│   ├── plant_model_agent.yaml        its system prompt and the template of every turn
│   ├── plant_model_metadata.yaml     the metadata agent (states, equations ... of a draft)
│   └── agent_notes.yaml              notes the app adds to a turn
├── search/
│   ├── rag.yaml                      search in the documents the user attached
│   └── websearch.yaml                web search (only when the user turns it on)
└── simulink/
    ├── upload_guidance.yaml          what the agent is told about a Simulink upload
    ├── chat_values.yaml              reading values of missing MATLAB variables from a message
    └── c2py_agent.yaml               the agent of the C-code route (readable dynamics, JSON card)
```

## Which prompt, when

| File · key | Used by | When |
|---|---|---|
| `chat/plant_model_agent` · `system_prompt`, `user_prompt_template` | `backend_core/agent.py` | every turn of the chat |
| `chat/plant_model_metadata` · `system_prompt`, `user_prompt_template` | `backend_core/agent.py` | after every new draft (its metadata) |
| `chat/agent_notes` · `repair_note` | `backend_core/agent.py` | the agent's reply was not valid JSON |
| `chat/agent_notes` · `force_draft_note` | `backend_core/agent.py` | the agent tried to finish too early |
| `chat/agent_notes` · `unusable_file` | `app.py` | a file the app cannot use was attached |
| `chat/agent_notes` · `converted_draft` | `app.py` | the draft came from a Simulink conversion |
| `search/rag` · `grounding_instructions` | `backend_core/rag.py` | a question is searched in the attached documents |
| `search/websearch` · `planner_*` | `backend_core/websearch.py` | web search on: is a search needed now? |
| `search/websearch` · `search_instructions` | `backend_core/websearch.py` | the web search itself |
| `simulink/upload_guidance` · `context_*`, `routes` | `simulink/intake.py` · `agent_context` | every turn while a Simulink model or C code waits |
| `simulink/upload_guidance` · `just_uploaded_*` | `simulink/intake.py` · `just_uploaded_note` | the turn in which the model arrives |
| `simulink/upload_guidance` · `values_*` | `simulink/intake.py` | the user sent values of missing MATLAB variables |
| `simulink/upload_guidance` · `*_diagram`, `context_signals`, `signals_failed` | `simulink/intake.py` | a model without root Inports / Outports: its inputs and outputs are chosen on its diagram |
| `simulink/upload_guidance` · `export_steps`, `manual_steps` | `simulink/intake.py` | the steps of route A / route B |
| `simulink/chat_values` · `system` | `simulink/direct/chat_values.py` | values given in words (the parser found no `name = value`) |
| `simulink/c2py_agent` · `head_*`, `common_*`, `rule7` | `simulink/c2py/agent.py` | rewriting the dynamics readable (C-code route) |
| `simulink/c2py_agent` · `annotate_*` | `simulink/c2py/agent.py` | big models: names and meanings only |
| `simulink/c2py_agent` · `user_template`, `flat_user`, `impure_*`, `retry_template` | `simulink/c2py/agent.py` | the user message of a rewrite, and after a rejected draft |
| `simulink/c2py_agent` · `describe_prompt*` | `simulink/c2py/agent.py` | naming the definitions of the JSON card |

## Placeholders

`{{name}}` is filled by `prompts.fill`; a placeholder without a value is an error, never sent to an
LLM. The c2py agent's prompts keep Python's `{name}` (filled with `str.format`), as in the
C_to_Python project they come from. Each file starts with comments on its keys.
