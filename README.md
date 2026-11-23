# JEV Agent — Mistral × TypeSafe Decision Pipeline

A transparent, multi-engine agent framework that combines **Mistral** for language synthesis with **JEV** (TypeSafe) for fast, calibrated decision-making. Built as a Streamlit playground to compare three distinct agentic architectures side by side.

## What makes this interesting

Most LLM agents are opaque black boxes: the model decides which tools to call, in what order, and when to stop — all hidden inside a single API call. This project makes every step visible and **swappable at runtime**:

| Engine | How it works | Latency profile |
|--------|-------------|-----------------|
| **JEV Pipeline** | TypeSafe's typed classifier routes to skill bundles (50–150ms), Mistral synthesizes | Ultra-fast routing, Mistral only for synthesis |
| **LLM Native** | Mistral picks its own tools via function calling, no external router | Single LLM loop, fully autonomous |
| **ReAct Planning** | Mistral plans a todo list first, then executes step-by-step with isolated skill contexts | Most transparent, best for complex multi-step tasks |

## Architecture

```
User query
    │
    ├─► [JEV Pipeline mode]
    │       JEV (TypeSafe) ──► select skill bundle
    │       JEV (TypeSafe) ──► select tool within bundle
    │       Python tool ──────► execute
    │       JEV (TypeSafe) ──► decide: call more tools / switch skill / synthesize
    │       Mistral ──────────► synthesize final answer
    │
    ├─► [LLM Native mode]
    │       Mistral (function calling) ──► pick & call tools autonomously
    │       Mistral ──────────────────────► stop when done
    │
    └─► [ReAct Planning mode]
            Mistral ──► produce step-by-step plan (JSON)
            For each step:
                load only that step's skill tools
                Mistral ──► reason + call tools
                unload skill context
            Mistral ──► synthesize from all step results
```

The key insight: **JEV handles routing decisions at ~100ms with calibrated confidence scores**, while Mistral handles open-ended language tasks. This separates concerns cleanly and makes routing auditable.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:
```bash
MISTRAL_API_KEY=your_key_here
TYPESAFE_API_KEY=your_key_here   # only needed for JEV Pipeline mode
```

Run the agent UI:
```bash
streamlit run agent_app.py
```

Or run the original Laya/JEV decision demo:
```bash
streamlit run app.py
```

## Skills & Tools

Skills are **named bundles of tools** that JEV activates based on query classification. The routing descriptions live in YAML — no code changes needed to add a new domain.

```
config/
  skills.yaml     ← skill bundles + their tool lists
  tools.yaml      ← tool descriptions (used by JEV for routing)
  agent_config.yaml ← engine parameters

agent/tools/
  calculator.py   ← math expressions
  datetime_tool.py ← date/time queries
  web_search.py   ← web search (stub)
  text_utils.py   ← text analysis
```

### Adding a new skill in 4 steps

1. **`config/skills.yaml`** — declare the skill and its tools
2. **`config/tools.yaml`** — describe each tool (JEV uses this for routing)
3. **`agent/tools/my_tool.py`** — implement `run(query, history) -> str`
4. **`agent/tools/__init__.py`** — register in `TOOL_REGISTRY`

The UI's **Skills & Tools** tab shows active skills live, and the **How to extend** tab walks through a full Finance skill example.

## LLM providers

The `LLM_PROVIDER` env var (or `config/agent_config.yaml`) switches the synthesis LLM:

| Provider | Env var | Default model |
|----------|---------|---------------|
| `mistral` (default) | `MISTRAL_API_KEY` | `mistral-small-latest` |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-haiku-4-5-20251001` |

## JEV — typed decision primitives

JEV (by [TypeSafe](https://typesafe.ai)) provides three decision types used throughout the pipeline:

- **`Choice`** — classify into one of N labeled options (with calibrated confidence)
- **`Score`** — ordinal scoring on a defined scale
- **`Noul`** — calibrated probability (0 → 1)

The original Laya/JEV demo in `app.py` shows these primitives directly: support triage, content moderation, LLM routing — all as typed decisions with no text generation.

## Running tests

```bash
pytest tests/
```

## Note on Laya (open-weights model)

The `app.py` demo also supports **Laya** (convaiinnovations, 421M params), a local open-weights model for typed decisions. Per public benchmarks, zero-shot Laya accuracy is below a majority-class baseline on its own benchmark — fine-tuning on your domain is required before production use. The latency/cost argument is real; the accuracy is not. See the [model page](https://huggingface.co/convaiinnovations/laya).
