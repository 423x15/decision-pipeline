# JEV-Driven Agent Pipeline — Design Spec

**Date:** 2026-09-21  
**Branche:** `feature/jev-agent-pipeline`

---

## Objectif

Construire un pipeline d'agent IA où JEV (TypeSafe) prend toutes les décisions de routing et d'orchestration. Mistral intervient uniquement pour la synthèse finale. L'interface Streamlit expose le chatbot, l'inspecteur de skills/tools et le debug des décisions JEV.

---

## Architecture

### Principe fondamental

JEV est le **cerveau décisionnel** du pipeline. Mistral est un **renderer** de réponse finale. Aucune logique de routing ne passe par Mistral.

```
User query
    │
    ▼
[JEV] skills_loader    → Choice: quel skill bundle charger ?
    │                     state: query + skills disponibles (skills.yaml)
    ▼
[JEV] tool_selector    → Choice: quel tool appeler ?
    │                     state: query + tools du bundle actif
    ▼
[TOOL] exécution       → résultat brut
    │
    ▼
[JEV] next_decision    → Choice: { call_another_tool | final_answer }
    │                     state: query + historique complet des appels
    │
    ├── call_another_tool  → retour à tool_selector (max 5 itérations)
    │
    └── final_answer
              │
              ▼
         [MISTRAL] synthèse(query + [(tool, input, result), ...]) → réponse finale
```

### Contraintes

- **5 itérations max** — si next_decision choisit call_another_tool 5 fois, on force final_answer
- **Mistral = synthèse uniquement** — reçoit (query, tool_history), génère la réponse, ne raisonne pas sur les tools
- **JEV = décisionnel** — 3 types d'appels JEV : skills_loader, tool_selector, next_decision
- **Tools = purs** — reçoivent des paramètres, retournent un résultat sérialisable

---

## Structure des fichiers

```
agent/
├── __init__.py
├── pipeline.py           # boucle principale, orchestre les 3 décisions JEV
├── skills_loader.py      # décision JEV : quel skill bundle activer
├── tool_selector.py      # décision JEV : quel tool appeler
├── decision_engine.py    # décision JEV : continue ou final
├── mistral_client.py     # appel Mistral API pour synthèse finale
└── tools/
    ├── __init__.py       # registre : { tool_name: callable }
    ├── calculator.py     # évalue expressions mathématiques (safe)
    ├── datetime_tool.py  # date/heure courante, calculs de durées
    ├── web_search.py     # stub résultats simulés
    └── text_utils.py     # comptage mots, extraction d'entités

config/
├── skills.yaml           # skill bundles + tools associés
└── tools.yaml            # définition détaillée de chaque tool

db/                       # créé à runtime
└── conversations.db      # SQLite

agent_app.py              # entrée Streamlit : 3 onglets
```

---

## Modules détaillés

### config/skills.yaml

```yaml
skills:
  - name: math_skills
    description: "Calculs, statistiques, conversions numériques"
    tools: [calculator, datetime_tool]

  - name: search_skills
    description: "Recherche d'informations, faits, actualités"
    tools: [web_search]

  - name: text_skills
    description: "Analyse de texte, comptage, résumé"
    tools: [text_utils]

  - name: general_skills
    description: "Tâche générale sans tool spécifique requis"
    tools: []
```

### config/tools.yaml

```yaml
tools:
  calculator:
    description: "Évalue une expression mathématique (ex: '1500 * 0.197')"
    input_hint: "expression mathématique en string"

  datetime_tool:
    description: "Retourne la date/heure courante ou calcule une durée"
    input_hint: "query temporelle en string"

  web_search:
    description: "Recherche des informations en temps réel sur un sujet"
    input_hint: "requête de recherche en string"

  text_utils:
    description: "Analyse un texte : compte les mots, extrait les entités"
    input_hint: "texte à analyser"
```

---

### agent/skills_loader.py

Entrée : query + skills disponibles depuis skills.yaml  
Décision JEV : Choice — quel skill bundle ?  
Sortie : liste des tool names actifs pour ce tour  

Appel JEV :
```python
response = client.system_one(
    state=f"Requête: {query}",
    questions={
        "skill": Choice(
            instructions="Quel skill bundle correspond le mieux à cette requête ?",
            criteria={s["name"]: s["description"] for s in skills}
        )
    }
)
# → response.choices["skill"].choice = "math_skills"
```

---

### agent/tool_selector.py

Entrée : query + tools actifs + historique des appels  
Décision JEV : Choice — quel tool ?  
Sortie : (tool_name, tool_input extrait de la query)  

L'input du tool est extrait de la query par une heuristique simple dans chaque tool.

---

### agent/decision_engine.py

Entrée : query + historique complet + iteration count  
Décision JEV : Choice(call_another_tool | final_answer) + Noul(sufficient_info)  
Sortie : "call_another_tool" ou "final_answer"  

Force final_answer si iteration >= 5.

---

### agent/pipeline.py

Orchestre la boucle. Émet des events via un générateur Python pour que l'UI affiche la trace inline au fil de l'exécution.

```python
@dataclass
class PipelineEvent:
    type: str           # "skill_loaded" | "tool_called" | "decision" | "final"
    data: dict
    jev_confidence: float | None
    iteration: int
    latency_ms: float
```

`run(query, jev_client, mistral_client)` → generator yielding PipelineEvent

---

### agent/mistral_client.py

Reçoit : query + liste [(tool_name, tool_input, tool_result)]  
Prompt minimaliste — Mistral formule, ne décide pas.

```python
prompt = (
    f"Question: {query}\n\n"
    f"Informations collectées:\n{context}\n\n"
    f"Réponds à la question en utilisant uniquement ces informations."
)
```

---

## Base de données SQLite

### Table conversations

```sql
CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    title TEXT
);
```

### Table messages

```sql
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tool_trace TEXT,       -- JSON des PipelineEvents
    created_at TEXT NOT NULL
);
```

---

## Interface Streamlit — agent_app.py

### Onglet 1 : Chat

- Sidebar : liste des conversations + bouton "Nouvelle conversation"
- Messages chronologiques
- Trace inline expandable sous chaque réponse assistant :
  ```
  🧠 JEV → math_skills [conf: 0.89] · 45ms
  🔧 JEV → calculator("1500 * 0.197") [conf: 0.94] · 38ms
  ✅ JEV → final_answer [conf: 0.91] · 41ms
  📝 Mistral synthesis · 620ms
  ```

### Onglet 2 : Skills & Tools

- Skills et tools chargés depuis YAML
- Éditeur YAML intégré + bouton "Recharger"
- Tableau : nom, description, bundle, nb appels en session

### Onglet 3 : Config & Debug

- Variables d'env actives (clés masquées)
- Dernier state JEV complet (JSON)
- Historique décisions session : tool, confiance, latence
- Distribution des tools appelés (bar chart)

---

## Dépendances à ajouter

```
mistralai>=1.0
```

.env :
```
TYPESAFE_API_KEY=...   # existant
MISTRAL_API_KEY=...    # nouveau
```

---

## Hors scope

- Authentification utilisateur
- Déploiement cloud
- Tools avec effets de bord réels
- Streaming token-par-token Mistral
