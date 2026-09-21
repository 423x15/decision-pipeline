# JEV-Driven Agent Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a JEV-driven agent pipeline where JEV makes all routing decisions and Mistral renders the final answer, with a Streamlit UI exposing chatbot, skills/tools inspector, and debug panel.

**Architecture:** JEV handles 3 decision types per loop turn (skills_loader → tool_selector → decision_engine). Tools execute and return results. After max 5 iterations or when JEV decides final_answer, Mistral synthesizes the full response from query + tool history.

**Tech Stack:** Python 3.12, typesafe-sdk, mistralai>=1.0, streamlit>=1.38, sqlite3 (stdlib), pyyaml, python-dotenv

**Spec:** `docs/superpowers/specs/2026-09-21-jev-agent-pipeline-design.md`

## Global Constraints

- Branch: `feature/jev-agent-pipeline`
- Max iterations per pipeline run: 5 (constant `MAX_ITERATIONS = 5`)
- JEV client: `TypeSafeClient` from `typesafe_sdk`, env var `TYPESAFE_API_KEY`
- Mistral client: `Mistral` from `mistralai`, env var `MISTRAL_API_KEY`, model `mistral-small-latest`
- All tool functions signature: `(query: str, history: list[dict]) -> str`
- `PipelineEvent` dataclass fields: `type`, `data`, `jev_confidence`, `iteration`, `latency_ms`
- SQLite DB path: `db/conversations.db` (created at runtime)
- All new files live under `agent/`, `config/`, `db/`, root `agent_app.py`

---

### Task 1: Config files + dependencies

**Files:**
- Create: `config/skills.yaml`
- Create: `config/tools.yaml`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `config/skills.yaml` loaded by `agent/skills_loader.py` (Task 3)
- Produces: `config/tools.yaml` loaded by `agent/tool_selector.py` (Task 4)

- [ ] **Step 1: Install mistralai**

```bash
pip install "mistralai>=1.0" "pyyaml>=6.0"
```

- [ ] **Step 2: Add to requirements.txt**

Append:
```
mistralai>=1.0
pyyaml>=6.0
```

- [ ] **Step 3: Create config/skills.yaml**

```bash
mkdir -p config
```

```yaml
# config/skills.yaml
skills:
  - name: math_skills
    description: "Calculs mathématiques, statistiques, conversions numériques, dates et durées"
    tools: [calculator, datetime_tool]

  - name: search_skills
    description: "Recherche d'informations, faits, actualités, définitions"
    tools: [web_search]

  - name: text_skills
    description: "Analyse de texte, comptage de mots, extraction d'entités"
    tools: [text_utils]

  - name: general_skills
    description: "Question générale ne nécessitant pas d'outil spécifique"
    tools: []
```

- [ ] **Step 4: Create config/tools.yaml**

```yaml
# config/tools.yaml
tools:
  calculator:
    description: "Évalue une expression mathématique (ex: '1500 * 0.197', '(42 + 8) / 5')"
    input_hint: "expression mathématique en string"

  datetime_tool:
    description: "Retourne la date et l'heure courantes, ou calcule une durée entre deux dates"
    input_hint: "query temporelle en langage naturel"

  web_search:
    description: "Recherche des informations sur un sujet (stub simulé, pas de clé requise)"
    input_hint: "requête de recherche en string"

  text_utils:
    description: "Analyse un texte : compte les mots, les phrases, extrait les mots-clés"
    input_hint: "texte à analyser"
```

- [ ] **Step 5: Commit**

```bash
git add requirements.txt config/skills.yaml config/tools.yaml
git commit -m "feat: add config YAML files and mistralai dependency"
```

---

### Task 2: Tool implementations + registry

**Files:**
- Create: `agent/__init__.py`
- Create: `agent/tools/__init__.py`
- Create: `agent/tools/calculator.py`
- Create: `agent/tools/datetime_tool.py`
- Create: `agent/tools/web_search.py`
- Create: `agent/tools/text_utils.py`
- Create: `tests/test_tools.py`

**Interfaces:**
- Produces: `TOOL_REGISTRY: dict[str, Callable[[str, list[dict]], str]]` from `agent/tools/__init__.py`
- Each tool: `run(query: str, history: list[dict]) -> str`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tools.py
import pytest
from agent.tools import TOOL_REGISTRY


def test_registry_has_all_tools():
    for name in ["calculator", "datetime_tool", "web_search", "text_utils"]:
        assert name in TOOL_REGISTRY


def test_calculator_basic():
    result = TOOL_REGISTRY["calculator"]("combien font 6 * 7", [])
    assert "42" in result


def test_calculator_expression_in_query():
    result = TOOL_REGISTRY["calculator"]("calcule 1500 * 0.197", [])
    assert "295.5" in result or "295,5" in result


def test_calculator_no_expression():
    result = TOOL_REGISTRY["calculator"]("quelle heure est-il", [])
    assert "expression" in result.lower() or "trouv" in result.lower()


def test_datetime_tool_returns_date():
    result = TOOL_REGISTRY["datetime_tool"]("quelle est la date aujourd'hui", [])
    assert "2026" in result or "septembre" in result.lower() or "09" in result


def test_web_search_returns_stub():
    result = TOOL_REGISTRY["web_search"]("prix bitcoin", [])
    assert len(result) > 10


def test_text_utils_word_count():
    result = TOOL_REGISTRY["text_utils"]("compte les mots dans: bonjour monde comment vas-tu", [])
    assert "mot" in result.lower() or any(c.isdigit() for c in result)
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_tools.py -v 2>&1 | head -20
```

Expected: ImportError

- [ ] **Step 3: Create agent/__init__.py**

```python
# agent/__init__.py
```

- [ ] **Step 4: Create agent/tools/calculator.py**

```python
# agent/tools/calculator.py
from __future__ import annotations
import re
import ast
import operator

_SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}

def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Opération non supportée: {ast.dump(node)}")

def _extract_expression(query: str) -> str | None:
    matches = re.findall(r"[\d\s\+\-\*\/\(\)\.\^]+", query)
    for m in sorted(matches, key=len, reverse=True):
        candidate = m.strip()
        if len(candidate) >= 3 and any(c in candidate for c in "+-*/"):
            return candidate
    return None

def run(query: str, history: list[dict]) -> str:
    expr = _extract_expression(query)
    if not expr:
        return "Aucune expression mathématique trouvée dans la requête."
    try:
        tree = ast.parse(expr.replace("^", "**"), mode="eval")
        result = _safe_eval(tree.body)
        if result == int(result):
            return f"{expr.strip()} = {int(result)}"
        return f"{expr.strip()} = {result:.6g}"
    except Exception as exc:
        return f"Erreur d'évaluation de '{expr}': {exc}"
```

- [ ] **Step 5: Create agent/tools/datetime_tool.py**

```python
# agent/tools/datetime_tool.py
from __future__ import annotations
from datetime import datetime

_MONTHS_FR = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre"
]

def run(query: str, history: list[dict]) -> str:
    now = datetime.now()
    q = query.lower()
    if any(w in q for w in ["heure", "time", "maintenant", "now"]):
        return f"Il est {now.strftime('%H:%M:%S')} le {now.day} {_MONTHS_FR[now.month-1]} {now.year}."
    if any(w in q for w in ["jour", "date", "aujourd", "today"]):
        return f"Nous sommes le {now.day} {_MONTHS_FR[now.month-1]} {now.year}."
    if any(w in q for w in ["année", "annee", "year"]):
        return f"Nous sommes en {now.year}."
    if any(w in q for w in ["mois", "month"]):
        return f"Nous sommes en {_MONTHS_FR[now.month-1]} {now.year}."
    return f"Date et heure : {now.strftime('%d/%m/%Y %H:%M:%S')}."
```

- [ ] **Step 6: Create agent/tools/web_search.py**

```python
# agent/tools/web_search.py
from __future__ import annotations
import hashlib

_STUBS = {
    "bitcoin": "Bitcoin (BTC) : cours simulé ~65 000 USD. Capitalisation ~1.3T USD. Variation 24h: +2.1%.",
    "tesla": "Tesla (TSLA) : cours simulé ~248 USD. P/E: 65. Variation 24h: -0.8%.",
    "météo": "Météo simulée : 22°C, partiellement nuageux, vent 15 km/h.",
    "france": "France : 68 millions d'habitants. Capitale: Paris. PIB: ~2 800 Mds EUR.",
    "python": "Python 3.12 : langage interprété créé par Guido van Rossum.",
    "ia": "Intelligence artificielle : domaine simulant l'intelligence humaine. LLMs, vision, robotique.",
}

def run(query: str, history: list[dict]) -> str:
    q = query.lower()
    for key, result in _STUBS.items():
        if key in q:
            return f"[Web Search Stub] {result}"
    seed = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)
    views = 1000 + (seed % 50000)
    return f"[Web Search Stub] {views:,} résultats simulés pour '{query}'. Informations générales disponibles."
```

- [ ] **Step 7: Create agent/tools/text_utils.py**

```python
# agent/tools/text_utils.py
from __future__ import annotations
import re
from collections import Counter

_STOP = {"le","la","les","de","du","des","un","une","et","en","à","au","aux",
         "est","je","tu","il","elle","nous","vous","ils","elles","que","qui",
         "ne","pas","se","ce","sa","son","dans","sur","par","pour","avec"}

def _extract_text(query: str) -> str:
    for sep in ["dans:", "dans :", "texte:", "texte :", "analyze:", "analyse:"]:
        if sep in query.lower():
            return query[query.lower().index(sep)+len(sep):].strip()
    return query

def run(query: str, history: list[dict]) -> str:
    text = _extract_text(query)
    words = re.findall(r"\b\w+\b", text.lower())
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    keywords = [w for w in words if w not in _STOP and len(w) > 2]
    top = [w for w, _ in Counter(keywords).most_common(5)]
    return (
        f"Analyse ({len(text)} caractères) :\n"
        f"- Mots : {len(words)}\n"
        f"- Phrases : {len(sentences)}\n"
        f"- Mots-clés : {', '.join(top) if top else 'aucun'}"
    )
```

- [ ] **Step 8: Create agent/tools/__init__.py**

```python
# agent/tools/__init__.py
from __future__ import annotations
from typing import Callable
from agent.tools import calculator, datetime_tool, web_search, text_utils

TOOL_REGISTRY: dict[str, Callable[[str, list[dict]], str]] = {
    "calculator": calculator.run,
    "datetime_tool": datetime_tool.run,
    "web_search": web_search.run,
    "text_utils": text_utils.run,
}
```

- [ ] **Step 9: Run tests**

```bash
python -m pytest tests/test_tools.py -v
```

Expected: all 7 PASS

- [ ] **Step 10: Commit**

```bash
git add agent/ tests/test_tools.py
git commit -m "feat: add tool implementations and registry"
```

---

### Task 3: JEV skills_loader

**Files:**
- Create: `agent/skills_loader.py`
- Create: `tests/test_skills_loader.py`

**Interfaces:**
- Produces: `load_skills_config(config_path: Path) -> list[dict]`
- Produces: `select_skill(query: str, skills: list[dict], jev_client) -> tuple[str, list[str], float, float]`
  - Returns `(skill_name, active_tool_names, confidence, latency_ms)`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_skills_loader.py
from pathlib import Path
from unittest.mock import MagicMock
from agent.skills_loader import load_skills_config, select_skill

SKILLS_YAML = Path("config/skills.yaml")

def test_load_skills_config_returns_list():
    skills = load_skills_config(SKILLS_YAML)
    assert isinstance(skills, list) and len(skills) > 0

def test_load_skills_config_has_required_fields():
    for s in load_skills_config(SKILLS_YAML):
        assert "name" in s and "description" in s and "tools" in s

def test_select_skill_returns_tuple():
    mock_client = MagicMock()
    mock_client.system_one.return_value = MagicMock(
        choices={"skill": MagicMock(choice="math_skills", confidence=0.9)}
    )
    skill_name, tools, confidence, latency = select_skill(
        "combien font 3 * 7", load_skills_config(SKILLS_YAML), mock_client
    )
    assert skill_name == "math_skills"
    assert isinstance(tools, list)
    assert 0.0 <= confidence <= 1.0
    assert latency >= 0.0

def test_select_skill_passes_query_in_state():
    mock_client = MagicMock()
    mock_client.system_one.return_value = MagicMock(
        choices={"skill": MagicMock(choice="search_skills", confidence=0.85)}
    )
    select_skill("cherche des infos sur python", load_skills_config(SKILLS_YAML), mock_client)
    state_arg = str(mock_client.system_one.call_args)
    assert "cherche des infos sur python" in state_arg
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_skills_loader.py -v 2>&1 | head -10
```

- [ ] **Step 3: Implement agent/skills_loader.py**

```python
# agent/skills_loader.py
from __future__ import annotations
import time
from pathlib import Path
import yaml
from typesafe_sdk import Choice

def load_skills_config(config_path: Path) -> list[dict]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["skills"]

def select_skill(
    query: str,
    skills: list[dict],
    jev_client,
) -> tuple[str, list[str], float, float]:
    criteria = {s["name"]: s["description"] for s in skills}
    t0 = time.perf_counter()
    response = jev_client.system_one(
        state=f"Requête utilisateur: {query}\nSkills disponibles: {', '.join(criteria)}",
        questions={
            "skill": Choice(
                instructions="Quel skill bundle correspond le mieux à cette requête ?",
                criteria=criteria,
            )
        },
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    chosen = response.choices["skill"].choice
    confidence = response.choices["skill"].confidence
    active_tools = next((s["tools"] for s in skills if s["name"] == chosen), [])
    return chosen, active_tools, confidence, latency_ms
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_skills_loader.py -v
```

Expected: all 4 PASS

- [ ] **Step 5: Commit**

```bash
git add agent/skills_loader.py tests/test_skills_loader.py
git commit -m "feat: add JEV skills_loader"
```

---

### Task 4: JEV tool_selector

**Files:**
- Create: `agent/tool_selector.py`
- Create: `tests/test_tool_selector.py`

**Interfaces:**
- Produces: `load_tools_config(config_path: Path) -> dict[str, dict]`
- Produces: `select_tool(query: str, active_tool_names: list[str], tools_config: dict[str, dict], history: list[dict], jev_client) -> tuple[str, float, float]`
  - Returns `(tool_name, confidence, latency_ms)`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tool_selector.py
from pathlib import Path
from unittest.mock import MagicMock
from agent.tool_selector import load_tools_config, select_tool

TOOLS_YAML = Path("config/tools.yaml")

def test_load_tools_config_returns_dict():
    tools = load_tools_config(TOOLS_YAML)
    assert isinstance(tools, dict) and "calculator" in tools

def test_load_tools_config_has_description():
    for name, meta in load_tools_config(TOOLS_YAML).items():
        assert "description" in meta

def test_select_tool_returns_tuple():
    mock_client = MagicMock()
    mock_client.system_one.return_value = MagicMock(
        choices={"tool": MagicMock(choice="calculator", confidence=0.95)}
    )
    tool_name, confidence, latency = select_tool(
        query="combien font 6 * 7",
        active_tool_names=["calculator", "datetime_tool"],
        tools_config=load_tools_config(TOOLS_YAML),
        history=[],
        jev_client=mock_client,
    )
    assert tool_name == "calculator"
    assert 0.0 <= confidence <= 1.0
    assert latency >= 0.0

def test_select_tool_includes_history_in_state():
    mock_client = MagicMock()
    mock_client.system_one.return_value = MagicMock(
        choices={"tool": MagicMock(choice="web_search", confidence=0.8)}
    )
    select_tool(
        query="test",
        active_tool_names=["web_search"],
        tools_config=load_tools_config(TOOLS_YAML),
        history=[{"tool": "calculator", "result": "42", "input": "6*7"}],
        jev_client=mock_client,
    )
    state_str = str(mock_client.system_one.call_args)
    assert "calculator" in state_str or "42" in state_str
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_tool_selector.py -v 2>&1 | head -10
```

- [ ] **Step 3: Implement agent/tool_selector.py**

```python
# agent/tool_selector.py
from __future__ import annotations
import time
from pathlib import Path
import yaml
from typesafe_sdk import Choice

def load_tools_config(config_path: Path) -> dict[str, dict]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["tools"]

def select_tool(
    query: str,
    active_tool_names: list[str],
    tools_config: dict[str, dict],
    history: list[dict],
    jev_client,
) -> tuple[str, float, float]:
    if not active_tool_names:
        return "web_search", 0.5, 0.0

    criteria = {
        name: tools_config[name]["description"]
        for name in active_tool_names
        if name in tools_config
    }
    history_summary = ""
    if history:
        lines = [f"  - {h['tool']}: {str(h['result'])[:80]}" for h in history[-3:]]
        history_summary = "\nHistorique récent:\n" + "\n".join(lines)

    t0 = time.perf_counter()
    response = jev_client.system_one(
        state=(
            f"Requête: {query}"
            f"{history_summary}\n"
            f"Tools disponibles: {', '.join(criteria)}"
        ),
        questions={
            "tool": Choice(
                instructions="Quel tool appeler pour progresser vers la réponse ?",
                criteria=criteria,
            )
        },
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    return response.choices["tool"].choice, response.choices["tool"].confidence, latency_ms
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_tool_selector.py -v
```

Expected: all 4 PASS

- [ ] **Step 5: Commit**

```bash
git add agent/tool_selector.py tests/test_tool_selector.py
git commit -m "feat: add JEV tool_selector"
```

---

### Task 5: JEV decision_engine

**Files:**
- Create: `agent/decision_engine.py`
- Create: `tests/test_decision_engine.py`

**Interfaces:**
- Produces: `MAX_ITERATIONS: int = 5`
- Produces: `decide(query: str, history: list[dict], iteration: int, jev_client) -> tuple[str, float, float]`
  - Returns `(decision, confidence, latency_ms)` where decision ∈ `{"call_another_tool", "final_answer"}`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_decision_engine.py
from unittest.mock import MagicMock
from agent.decision_engine import decide, MAX_ITERATIONS

def test_max_iterations_is_5():
    assert MAX_ITERATIONS == 5

def test_decide_forces_final_at_max_iterations():
    mock_client = MagicMock()
    decision, confidence, latency = decide("test", [{"tool": "x", "result": "y"}], 5, mock_client)
    assert decision == "final_answer"
    mock_client.system_one.assert_not_called()

def test_decide_returns_jev_choice_before_max():
    mock_client = MagicMock()
    mock_client.system_one.return_value = MagicMock(
        choices={"next": MagicMock(choice="final_answer", confidence=0.92)},
        nouls={"sufficient_info": MagicMock(probability=0.88)},
    )
    decision, confidence, latency = decide(
        "quelle est la date?",
        [{"tool": "datetime_tool", "result": "2026-09-21", "input": "date"}],
        1, mock_client,
    )
    assert decision == "final_answer"
    assert confidence == 0.92

def test_decide_call_another_tool():
    mock_client = MagicMock()
    mock_client.system_one.return_value = MagicMock(
        choices={"next": MagicMock(choice="call_another_tool", confidence=0.75)},
        nouls={"sufficient_info": MagicMock(probability=0.3)},
    )
    decision, _, _ = decide(
        "combien font 3*7 et quelle heure",
        [{"tool": "calculator", "result": "21", "input": "3*7"}],
        1, mock_client,
    )
    assert decision == "call_another_tool"
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_decision_engine.py -v 2>&1 | head -10
```

- [ ] **Step 3: Implement agent/decision_engine.py**

```python
# agent/decision_engine.py
from __future__ import annotations
import time
from typesafe_sdk import Choice, Noul

MAX_ITERATIONS = 5

def decide(
    query: str,
    history: list[dict],
    iteration: int,
    jev_client,
) -> tuple[str, float, float]:
    if iteration >= MAX_ITERATIONS:
        return "final_answer", 1.0, 0.0

    results_summary = "\n".join(
        f"  [{i+1}] {h['tool']}({h.get('input','')[:40]}) → {str(h['result'])[:120]}"
        for i, h in enumerate(history)
    )
    t0 = time.perf_counter()
    response = jev_client.system_one(
        state=(
            f"Requête: {query}\n\n"
            f"Résultats ({len(history)} tool(s)):\n{results_summary or '  (aucun)'}\n\n"
            f"Itération: {iteration + 1}/{MAX_ITERATIONS}"
        ),
        questions={
            "next": Choice(
                instructions="Faut-il appeler un autre tool ou a-t-on assez d'informations ?",
                criteria={
                    "call_another_tool": "Des informations manquent encore pour répondre complètement",
                    "final_answer": "Les informations collectées sont suffisantes pour répondre",
                },
            ),
            "sufficient_info": Noul(
                instructions="Les résultats permettent-ils une réponse complète et précise ?"
            ),
        },
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    return response.choices["next"].choice, response.choices["next"].confidence, latency_ms
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_decision_engine.py -v
```

Expected: all 4 PASS

- [ ] **Step 5: Commit**

```bash
git add agent/decision_engine.py tests/test_decision_engine.py
git commit -m "feat: add JEV decision_engine with MAX_ITERATIONS=5"
```

---

### Task 6: Mistral synthesis client

**Files:**
- Create: `agent/mistral_client.py`
- Create: `tests/test_mistral_client.py`

**Interfaces:**
- Produces: `build_synthesis_prompt(query: str, history: list[dict]) -> str`
- Produces: `MistralClient(api_key: str | None, model: str)` with `.synthesize(query: str, history: list[dict]) -> str`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_mistral_client.py
from unittest.mock import MagicMock
from agent.mistral_client import MistralClient, build_synthesis_prompt

def test_build_synthesis_prompt_contains_query():
    prompt = build_synthesis_prompt(
        "combien font 6 * 7",
        [{"tool": "calculator", "input": "6 * 7", "result": "6 * 7 = 42"}],
    )
    assert "6 * 7" in prompt and "42" in prompt

def test_build_synthesis_prompt_empty_history():
    prompt = build_synthesis_prompt("quelle heure", [])
    assert "quelle heure" in prompt
    assert "Aucun" in prompt or "aucun" in prompt

def test_mistral_client_synthesize_calls_api():
    mock_mistral = MagicMock()
    mock_mistral.chat.complete.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="La réponse est 42."))]
    )
    client = MistralClient.__new__(MistralClient)
    client._client = mock_mistral
    client.model = "mistral-small-latest"

    result = client.synthesize("combien font 6 * 7", [{"tool": "calculator", "input": "6*7", "result": "42"}])
    assert result == "La réponse est 42."
    mock_mistral.chat.complete.assert_called_once()
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_mistral_client.py -v 2>&1 | head -10
```

- [ ] **Step 3: Implement agent/mistral_client.py**

```python
# agent/mistral_client.py
from __future__ import annotations
import os
from mistralai import Mistral

DEFAULT_MODEL = "mistral-small-latest"

def build_synthesis_prompt(query: str, history: list[dict]) -> str:
    if history:
        context = "\n".join(
            f"- {h['tool']}({h.get('input','')[:60]}) → {h['result']}"
            for h in history
        )
    else:
        context = "Aucun outil n'a été appelé."
    return (
        f"Question: {query}\n\n"
        f"Informations collectées:\n{context}\n\n"
        f"Réponds de manière concise en te basant uniquement sur ces informations."
    )

class MistralClient:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        key = api_key or os.environ.get("MISTRAL_API_KEY", "")
        if not key:
            raise ValueError("MISTRAL_API_KEY manquant. Ajoute-le dans .env")
        self._client = Mistral(api_key=key)
        self.model = model

    def synthesize(self, query: str, history: list[dict]) -> str:
        response = self._client.chat.complete(
            model=self.model,
            messages=[{"role": "user", "content": build_synthesis_prompt(query, history)}],
        )
        return response.choices[0].message.content
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_mistral_client.py -v
```

Expected: all 3 PASS

- [ ] **Step 5: Commit**

```bash
git add agent/mistral_client.py tests/test_mistral_client.py
git commit -m "feat: add Mistral synthesis client"
```

---

### Task 7: Pipeline orchestrator

**Files:**
- Create: `agent/pipeline.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `select_skill`, `select_tool`, `decide`, `MistralClient`, `TOOL_REGISTRY`
- Produces: `PipelineEvent` dataclass with fields `type: str`, `data: dict`, `jev_confidence: float | None`, `iteration: int`, `latency_ms: float`
- Produces: `run(query: str, jev_client, mistral_client, skills_config: list[dict], tools_config: dict[str, dict]) -> Generator[PipelineEvent, None, None]`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_pipeline.py
from pathlib import Path
from unittest.mock import MagicMock
from agent.pipeline import PipelineEvent, run
from agent.skills_loader import load_skills_config
from agent.tool_selector import load_tools_config


def _make_jev(skill="math_skills", tool="calculator", decision="final_answer"):
    client = MagicMock()
    call_count = [0]
    def side_effect(state, questions):
        call_count[0] += 1
        resp = MagicMock()
        keys = list(questions.keys())
        if "skill" in keys:
            resp.choices = {"skill": MagicMock(choice=skill, confidence=0.9)}
        elif "tool" in keys:
            resp.choices = {"tool": MagicMock(choice=tool, confidence=0.9)}
        else:
            resp.choices = {"next": MagicMock(choice=decision, confidence=0.9)}
            resp.nouls = {"sufficient_info": MagicMock(probability=0.8)}
        return resp
    client.system_one.side_effect = side_effect
    return client


def test_pipeline_event_dataclass():
    e = PipelineEvent(type="skill_loaded", data={"skill": "math_skills"},
                      jev_confidence=0.9, iteration=0, latency_ms=45.0)
    assert e.type == "skill_loaded"
    assert e.jev_confidence == 0.9


def test_run_yields_all_event_types():
    skills = load_skills_config(Path("config/skills.yaml"))
    tools_cfg = load_tools_config(Path("config/tools.yaml"))
    jev = _make_jev()
    mistral = MagicMock()
    mistral.synthesize.return_value = "La réponse est 42."

    events = list(run("combien font 6*7", jev, mistral, skills, tools_cfg))
    types = {e.type for e in events}
    assert "skill_loaded" in types
    assert "tool_called" in types
    assert "decision" in types
    assert "final" in types


def test_run_final_event_has_response():
    skills = load_skills_config(Path("config/skills.yaml"))
    tools_cfg = load_tools_config(Path("config/tools.yaml"))
    jev = _make_jev()
    mistral = MagicMock()
    mistral.synthesize.return_value = "Réponse finale."

    events = list(run("test", jev, mistral, skills, tools_cfg))
    final = next(e for e in events if e.type == "final")
    assert final.data["response"] == "Réponse finale."
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_pipeline.py -v 2>&1 | head -15
```

- [ ] **Step 3: Implement agent/pipeline.py**

```python
# agent/pipeline.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

from agent.skills_loader import select_skill
from agent.tool_selector import select_tool
from agent.decision_engine import decide, MAX_ITERATIONS
from agent.tools import TOOL_REGISTRY


@dataclass
class PipelineEvent:
    type: str
    data: dict = field(default_factory=dict)
    jev_confidence: float | None = None
    iteration: int = 0
    latency_ms: float = 0.0


def run(
    query: str,
    jev_client,
    mistral_client,
    skills_config: list[dict],
    tools_config: dict[str, dict],
) -> Generator[PipelineEvent, None, None]:
    history: list[dict] = []

    skill_name, active_tools, skill_conf, skill_lat = select_skill(query, skills_config, jev_client)
    yield PipelineEvent(
        type="skill_loaded",
        data={"skill": skill_name, "tools": active_tools},
        jev_confidence=skill_conf,
        iteration=0,
        latency_ms=skill_lat,
    )

    for iteration in range(MAX_ITERATIONS):
        tool_name, tool_conf, tool_sel_lat = select_tool(
            query, active_tools, tools_config, history, jev_client
        )
        tool_fn = TOOL_REGISTRY.get(tool_name)
        tool_result = tool_fn(query, history) if tool_fn else f"Tool '{tool_name}' non trouvé."

        history.append({"tool": tool_name, "input": query, "result": tool_result})
        yield PipelineEvent(
            type="tool_called",
            data={"tool": tool_name, "result": tool_result},
            jev_confidence=tool_conf,
            iteration=iteration,
            latency_ms=tool_sel_lat,
        )

        decision, dec_conf, dec_lat = decide(query, history, iteration, jev_client)
        yield PipelineEvent(
            type="decision",
            data={"decision": decision},
            jev_confidence=dec_conf,
            iteration=iteration,
            latency_ms=dec_lat,
        )
        if decision == "final_answer":
            break

    final_response = mistral_client.synthesize(query, history)
    yield PipelineEvent(
        type="final",
        data={"response": final_response, "tool_history": history},
        jev_confidence=None,
        iteration=len(history),
        latency_ms=0.0,
    )
```

- [ ] **Step 4: Run all tests**

```bash
python -m pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add agent/pipeline.py tests/test_pipeline.py
git commit -m "feat: add pipeline orchestrator with PipelineEvent generator"
```

---

### Task 8: SQLite database layer

**Files:**
- Create: `agent/db.py`
- Create: `tests/test_db.py`

**Interfaces:**
- Produces: `Database(db_path: Path)` with methods:
  - `create_conversation(title: str) -> str` (uuid4 string)
  - `list_conversations() -> list[dict]` (each dict has `id`, `title`, `created_at`)
  - `save_message(conv_id: str, role: str, content: str, tool_trace: list[dict] | None) -> None`
  - `load_messages(conv_id: str) -> list[dict]` (each dict has `role`, `content`, `tool_trace`, `created_at`)
  - `delete_conversation(conv_id: str) -> None`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_db.py
import pytest
from pathlib import Path
from agent.db import Database


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "test.db")


def test_create_conversation_returns_uuid(db):
    conv_id = db.create_conversation("Test")
    assert isinstance(conv_id, str) and len(conv_id) == 36


def test_list_conversations_empty(db):
    assert db.list_conversations() == []


def test_list_conversations_after_create(db):
    db.create_conversation("Conv 1")
    db.create_conversation("Conv 2")
    convs = db.list_conversations()
    assert len(convs) == 2
    assert all("id" in c and "title" in c and "created_at" in c for c in convs)


def test_save_and_load_messages(db):
    conv_id = db.create_conversation("Test")
    db.save_message(conv_id, "user", "Bonjour", None)
    db.save_message(conv_id, "assistant", "Réponse", [{"type": "tool_called"}])
    msgs = db.load_messages(conv_id)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["tool_trace"] == [{"type": "tool_called"}]


def test_delete_conversation(db):
    conv_id = db.create_conversation("To delete")
    db.save_message(conv_id, "user", "msg", None)
    db.delete_conversation(conv_id)
    assert db.list_conversations() == []
    assert db.load_messages(conv_id) == []
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_db.py -v 2>&1 | head -10
```

- [ ] **Step 3: Implement agent/db.py**

```python
# agent/db.py
from __future__ import annotations
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

_CREATE_CONVERSATIONS = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    title TEXT
);
"""
_CREATE_MESSAGES = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tool_trace TEXT,
    created_at TEXT NOT NULL
);
"""

class Database:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = str(db_path)
        with self._connect() as conn:
            conn.execute(_CREATE_CONVERSATIONS)
            conn.execute(_CREATE_MESSAGES)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_conversation(self, title: str) -> str:
        conv_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, created_at, title) VALUES (?, ?, ?)",
                (conv_id, self._now(), title[:80]),
            )
        return conv_id

    def list_conversations(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at FROM conversations ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def save_message(self, conversation_id: str, role: str, content: str, tool_trace: list[dict] | None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, tool_trace, created_at) VALUES (?,?,?,?,?)",
                (conversation_id, role, content, json.dumps(tool_trace) if tool_trace else None, self._now()),
            )

    def load_messages(self, conversation_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content, tool_trace, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY id ASC", (conversation_id,)
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["tool_trace"] = json.loads(d["tool_trace"]) if d["tool_trace"] else None
            result.append(d)
        return result

    def delete_conversation(self, conversation_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_db.py -v
```

Expected: all 5 PASS

- [ ] **Step 5: Run full suite**

```bash
python -m pytest tests/ -v
```

- [ ] **Step 6: Commit**

```bash
git add agent/db.py tests/test_db.py
git commit -m "feat: add SQLite database layer"
```

---

### Task 9: Streamlit UI — agent_app.py

**Files:**
- Create: `agent_app.py`

**Interfaces:**
- Consumes: `Database`, `run` (pipeline), `PipelineEvent`, `MistralClient`
- Consumes: `load_skills_config(Path("config/skills.yaml"))`, `load_tools_config(Path("config/tools.yaml"))`
- Consumes: `get_client` and `default_api_key` from `jev_client`

- [ ] **Step 1: Add MISTRAL_API_KEY to .env**

Add line to `.env`:
```
MISTRAL_API_KEY=your_mistral_api_key_here
```

- [ ] **Step 2: Create agent_app.py**

```python
# agent_app.py
from __future__ import annotations
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agent.db import Database
from agent.mistral_client import MistralClient
from agent.pipeline import PipelineEvent, run as pipeline_run
from agent.skills_loader import load_skills_config
from agent.tool_selector import load_tools_config
from jev_client import get_client as get_jev_client, default_api_key

DB_PATH = Path("db/conversations.db")
SKILLS_YAML = Path("config/skills.yaml")
TOOLS_YAML = Path("config/tools.yaml")


def _fmt_event(event: PipelineEvent) -> str:
    conf = f"[conf: {event.jev_confidence:.2f}]" if event.jev_confidence is not None else ""
    lat = f"· {event.latency_ms:.0f}ms"
    if event.type == "skill_loaded":
        tools = ", ".join(event.data.get("tools", [])) or "—"
        return f"🧠 JEV → {event.data['skill']} {conf} {lat}  _(tools: {tools})_"
    if event.type == "tool_called":
        preview = str(event.data.get("result", ""))[:60]
        return f"🔧 JEV → {event.data['tool']} {conf} {lat}  → `{preview}`"
    if event.type == "decision":
        icon = "✅" if event.data["decision"] == "final_answer" else "🔄"
        return f"{icon} JEV → {event.data['decision']} {conf} {lat}"
    if event.type == "final":
        return f"📝 Mistral synthesis {lat}"
    return f"• {event.type} {lat}"


@st.cache_resource
def get_db() -> Database:
    return Database(DB_PATH)


@st.cache_resource
def get_configs():
    return load_skills_config(SKILLS_YAML), load_tools_config(TOOLS_YAML)


def render_chat(db: Database, jev_client, mistral_client, skills_config, tools_config):
    with st.sidebar:
        st.subheader("Conversations")
        if st.button("➕ Nouvelle", width="stretch"):
            st.session_state["active_conv_id"] = None
            st.rerun()
        for conv in db.list_conversations():
            cols = st.columns([5, 1])
            label = (conv["title"] or conv["id"][:8])[:40]
            if cols[0].button(label, key=f"c_{conv['id']}", width="stretch"):
                st.session_state["active_conv_id"] = conv["id"]
                st.rerun()
            if cols[1].button("🗑", key=f"d_{conv['id']}"):
                db.delete_conversation(conv["id"])
                if st.session_state.get("active_conv_id") == conv["id"]:
                    st.session_state["active_conv_id"] = None
                st.rerun()

    conv_id = st.session_state.get("active_conv_id")
    if conv_id:
        for msg in db.load_messages(conv_id):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("tool_trace"):
                    with st.expander("🔍 Trace JEV", expanded=False):
                        for item in msg["tool_trace"]:
                            st.markdown(f"`{_fmt_event(PipelineEvent(**item))}`")

    user_input = st.chat_input("Posez votre question...")
    if not user_input:
        return
    if not jev_client:
        st.error("TYPESAFE_API_KEY manquante dans .env")
        return
    if not mistral_client:
        st.error("MISTRAL_API_KEY manquante dans .env")
        return

    if not conv_id:
        conv_id = db.create_conversation(user_input[:50])
        st.session_state["active_conv_id"] = conv_id

    db.save_message(conv_id, "user", user_input, None)
    with st.chat_message("user"):
        st.markdown(user_input)

    events: list[PipelineEvent] = []
    final_response = ""
    with st.chat_message("assistant"):
        trace_ph = st.empty()
        resp_ph = st.empty()
        trace_lines: list[str] = []

        for event in pipeline_run(user_input, jev_client, mistral_client, skills_config, tools_config):
            events.append(event)
            if event.type != "final":
                trace_lines.append(_fmt_event(event))
                trace_ph.markdown("\n\n".join(f"`{l}`" for l in trace_lines))
            else:
                final_response = event.data.get("response", "")
                resp_ph.markdown(final_response)
                trace_ph.empty()

        with st.expander("🔍 Trace JEV", expanded=False):
            for line in trace_lines:
                st.markdown(f"`{line}`")
        st.markdown(final_response)

    trace_dicts = [
        {"type": e.type, "data": e.data, "jev_confidence": e.jev_confidence,
         "iteration": e.iteration, "latency_ms": e.latency_ms}
        for e in events
    ]
    db.save_message(conv_id, "assistant", final_response, trace_dicts)
    st.rerun()


def render_skills_tools(skills_config, tools_config):
    st.subheader("Skills actifs")
    st.dataframe(pd.DataFrame([
        {"Skill": s["name"], "Description": s["description"],
         "Tools": ", ".join(s["tools"]) if s["tools"] else "—"}
        for s in skills_config
    ]), hide_index=True)

    st.divider()
    st.subheader("Tools disponibles")
    st.dataframe(pd.DataFrame([
        {"Tool": name, "Description": meta["description"], "Input hint": meta.get("input_hint", "")}
        for name, meta in tools_config.items()
    ]), hide_index=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("skills.yaml")
        content = SKILLS_YAML.read_text() if SKILLS_YAML.exists() else ""
        new_content = st.text_area("Éditer", value=content, height=250, key="skills_editor")
        if st.button("💾 Sauvegarder skills.yaml"):
            SKILLS_YAML.write_text(new_content)
            st.cache_resource.clear()
            st.rerun()
    with col2:
        st.subheader("tools.yaml")
        content = TOOLS_YAML.read_text() if TOOLS_YAML.exists() else ""
        new_content = st.text_area("Éditer", value=content, height=250, key="tools_editor")
        if st.button("💾 Sauvegarder tools.yaml"):
            TOOLS_YAML.write_text(new_content)
            st.cache_resource.clear()
            st.rerun()


def render_debug():
    st.subheader("Variables d'environnement")
    for key in ["TYPESAFE_API_KEY", "MISTRAL_API_KEY"]:
        val = os.environ.get(key, "")
        status = "✅ définie" if val else "❌ manquante"
        masked = (val[:4] + "****" + val[-4:]) if len(val) > 8 else "****"
        st.text(f"{key}: {status}  {masked if val else ''}")

    st.divider()
    decisions = st.session_state.get("session_decisions", [])
    if not decisions:
        st.info("Lance une requête depuis Chat pour voir les décisions ici.")
        return
    df = pd.DataFrame(decisions)
    st.subheader("Décisions JEV — session")
    st.dataframe(df, hide_index=True)
    if "tool" in df.columns:
        counts = df[df["type"] == "tool_called"]["tool"].value_counts()
        if not counts.empty:
            st.bar_chart(counts)


def main():
    st.set_page_config(page_title="JEV Agent", page_icon="🤖", layout="wide")

    for key, val in [("active_conv_id", None), ("session_decisions", [])]:
        if key not in st.session_state:
            st.session_state[key] = val

    st.title("🤖 JEV Agent Pipeline")
    st.caption("JEV orchestre · Mistral synthétise · Trace inline")

    jev_client = None
    mistral_client = None
    api_key = default_api_key()
    if api_key:
        try:
            jev_client = get_jev_client(api_key)
        except Exception:
            pass
    mistral_key = os.environ.get("MISTRAL_API_KEY", "")
    if mistral_key:
        try:
            mistral_client = MistralClient(api_key=mistral_key)
        except Exception:
            pass

    db = get_db()
    skills_config, tools_config = get_configs()

    tab_chat, tab_skills, tab_debug = st.tabs(["💬 Chat", "🛠 Skills & Tools", "⚙️ Config & Debug"])
    with tab_chat:
        render_chat(db, jev_client, mistral_client, skills_config, tools_config)
    with tab_skills:
        render_skills_tools(skills_config, tools_config)
    with tab_debug:
        render_debug()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 4: Smoke-test the app**

```bash
streamlit run agent_app.py
```

Vérifier :
- Onglet Chat : saisir "combien font 6 * 7" → trace JEV inline + réponse Mistral
- Onglet Skills & Tools : tableaux visibles, éditeur YAML fonctionnel
- Onglet Config & Debug : clés masquées affichées

- [ ] **Step 5: Final commit**

```bash
git add agent_app.py
git commit -m "feat: add Streamlit agent UI with chat, skills inspector, debug panel"
```
