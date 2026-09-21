# Ajouter des Tools et des Skills

## Vue d'ensemble

```
config/skills.yaml       ← JEV choisit quel bundle charger
config/tools.yaml        ← JEV choisit quel tool appeler
agent/tools/             ← implémentation Python de chaque tool
agent/tools/__init__.py  ← registre nom → fonction
```

---

## Créer un nouveau Tool

### 1. Implémenter — `agent/tools/mon_tool.py`

```python
def run(query: str, history: list[dict]) -> str:
    """
    query   : la question brute de l'utilisateur
    history : appels précédents [{"tool": ..., "input": ..., "result": ...}]
    retour  : résultat sérialisable en string
    """
    # logique ici
    return "résultat"
```

**Règles :**
- Signature toujours `run(query: str, history: list[dict]) -> str`
- Retourner une string (passée directement au LLM)
- Pas d'appel réseau réel sauf si c'est l'objet du tool (stubs en dev)

**Exemple — tool météo :**

```python
import httpx

def run(query: str, history: list[dict]) -> str:
    city = _extract_city(query) or "Paris"
    try:
        resp = httpx.get(f"https://wttr.in/{city}?format=3", timeout=5)
        return resp.text
    except Exception as exc:
        return f"Erreur météo : {exc}"

def _extract_city(query: str) -> str | None:
    words = query.lower().split()
    triggers = ["à", "a", "pour", "in"]
    for i, w in enumerate(words):
        if w in triggers and i + 1 < len(words):
            return words[i + 1].capitalize()
    return None
```

### 2. Enregistrer — `agent/tools/__init__.py`

```python
from agent.tools import calculator, datetime_tool, web_search, text_utils, mon_tool  # ← ajouter

TOOL_REGISTRY: dict[str, Callable[[str, list[dict]], str]] = {
    "calculator":    calculator.run,
    "datetime_tool": datetime_tool.run,
    "web_search":    web_search.run,
    "text_utils":    text_utils.run,
    "mon_tool":      mon_tool.run,   # ← ajouter
}
```

### 3. Décrire — `config/tools.yaml`

```yaml
tools:
  # ... tools existants ...
  mon_tool:
    description: "Description précise pour JEV — ce que le tool fait et quand l'utiliser"
    input_hint: "type d'entrée attendu (ex: question en langage naturel)"
```

**La `description` est critique** — JEV l'utilise pour discriminer entre tools. Sois précis et distinctif.

---

## Créer un nouveau Skill

Un skill est un **bundle de tools** activé par JEV selon la nature de la requête.

### Ajouter dans `config/skills.yaml`

```yaml
skills:
  # ... skills existants ...
  - name: weather_skills
    description: "Météo, prévisions, conditions climatiques, température"
    tools:
      - mon_tool
```

**Règles :**
- `name` : snake_case, unique
- `description` : phrase précise avec mots-clés déclencheurs — JEV s'en sert pour le matching
- `tools` : noms présents dans `TOOL_REGISTRY` et `tools.yaml`
- `tools: []` → JEV saute la boucle, OpenAI répond directement (utile pour conversation générale)

---

## Bonnes descriptions pour JEV

La qualité des descriptions détermine la précision du routing.

| ❌ Mauvais | ✅ Bon |
|-----------|--------|
| `"Fait des choses"` | `"Convertit des devises, calcule des taux de change"` |
| `"Recherche"` | `"Recherche temps réel : prix boursiers, actualités, données live"` |
| `"Gère le texte"` | `"Analyse un texte : compte mots/phrases, extrait mots-clés"` |

Si deux descriptions se ressemblent, JEV hésitera. Différencie par le domaine et les données typiques.

---

## Tester un tool sans l'UI

```python
# REPL ou script
from agent.tools.mon_tool import run
print(run("météo à Lyon", []))
```

Pytest minimal :

```python
# tests/test_mon_tool.py
from agent.tools.mon_tool import run

def test_run_returns_string():
    result = run("météo à Paris", [])
    assert isinstance(result, str) and len(result) > 0
```

---

## Voir les I/O en temps réel dans l'UI

Dans l'onglet **💬 Chat**, chaque réponse a un expander **🔍 Trace JEV** :

- `📥 nom_du_tool — I/O complet` → query reçue + résultat brut retourné par le tool
- `📤 Prompt envoyé au LLM` → prompt exact envoyé à OpenAI avec tout le contexte tools

Utile pour débugger pourquoi JEV choisit le mauvais tool ou pourquoi la réponse finale est incorrecte.

---

## Checklist

- [ ] `agent/tools/mon_tool.py` — fonction `run(query, history) -> str`
- [ ] `agent/tools/__init__.py` — import + entrée dans `TOOL_REGISTRY`
- [ ] `config/tools.yaml` — `description` + `input_hint`
- [ ] `config/skills.yaml` — ajout dans un skill existant ou nouveau skill
- [ ] `tests/test_mon_tool.py` — au moins `test_run_returns_string`
- [ ] Test manuel : poser une question dans l'UI, vérifier la trace 🔍
