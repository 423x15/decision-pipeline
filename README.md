# Laya Playground

Petit dashboard Streamlit pour explorer des moteurs de décision **typée**
(`choice` / `score` / `noul` → probabilités calibrées, pas de génération de
texte). Deux providers interchangeables derrière la même interface :

- **[Laya](https://huggingface.co/convaiinnovations/laya)** (convaiinnovations) — open-weights, 421M params, tourne 100% en local
- **[Jev](https://typesafe.ai)** (TypeSafe) — API hébergée, mêmes primitives (`Choice`/`Score`/`Noul`)

## Démarrage rapide (mode démo, sans rien installer côté modèle)

```bash
python -m venv .venv && source .venv/bin/activate   # optionnel
pip install streamlit pandas
streamlit run app.py
```

Ouvre `http://localhost:8501`, laisse **"Démo simulée"** sélectionné dans la
barre latérale, choisis un scénario (triage support, modération de contenu,
routage de modèle LLM...) et clique sur **Lancer la décision**. Les réponses
sont simulées (déterministes, basées sur le texte saisi) pour explorer
l'interface immédiatement.

## Utiliser Laya (local)

```bash
pip install -r requirements.txt
```

Dans la barre latérale, passe sur **"Laya (modèle local)"** et clique sur
**Charger le modèle Laya**. Le premier chargement télécharge les poids depuis
Hugging Face (~421M paramètres) — connexion internet nécessaire à ce
moment-là seulement. Ensuite le modèle tourne 100% en local (CPU ou GPU si
dispo). Si le chargement semble bloqué : `USE_TF=0 streamlit run app.py`.

## Utiliser Jev (API TypeSafe)

```bash
pip install typesafe-sdk   # déjà inclus dans requirements.txt
```

Crée une clé API sur [console.typesafe.ai](https://console.typesafe.ai/),
puis dans la barre latérale passe sur **"Jev (API TypeSafe)"** et colle la
clé (ou exporte `TYPESAFE_API_KEY` avant de lancer Streamlit).

⚠️ **Contrairement à Laya, ce mode envoie l'état et les questions au cloud de
TypeSafe** — ce n'est plus local. Ne l'utilise pas avec des données
sensibles/clients réelles sans avoir vérifié leurs conditions de traitement
des données.

## Ce que fait l'appli

- 4 scénarios prêts à l'emploi (triage support, modération, routage LLM) +
  un mode **Personnalisé** où tu édites toi-même les questions (clé, type
  `choice`/`score`/`noul`, instructions, critères) dans un tableau éditable.
- Visualisation dédiée par type de question, identique quel que soit le provider :
  - `choice` → barres de probabilité par option, gagnant mis en valeur
  - `score` → jauge/curseur sur l'échelle ordinale définie
  - `noul` → jauge de probabilité calibrée (vert → jaune → rouge)
- Latence de la passe d'inférence affichée en gros.
- Historique des runs (avec le provider utilisé) dans la barre latérale.
- Réponse JSON brute disponible dans un expander pour debug.

## Note sur le format exact des réponses

- **Jev** : le format de requête (`Choice`/`Score`/`Noul`, `state`/`questions`)
  et les champs de réponse (`.choice`, `.score`, `.noul`) viennent de la doc
  officielle SDK Python (`docs.typesafe.ai/sdk/python`) — fiable.
- **Laya** : le champ exact renvoyé par `agent.predict(...)` (ex.
  `probabilities` vs `scores` vs `distribution`) a été reconstitué à partir
  de la description du package sur PyPI, faute d'accès direct à
  `huggingface.co` depuis l'environnement où ce projet a été créé — à
  vérifier une fois le vrai modèle chargé.

Dans les deux cas, `ui.py` essaie plusieurs noms de clés courants et affiche
toujours le JSON brut en repli (expander "Réponse brute (JSON)") — si un
champ ne s'affiche pas correctement, regarde la structure dedans et ajuste
`_get_first(...)` dans `ui.py` en conséquence (une seule ligne par type de
question).

## ⚠️ Sur la qualité réelle des modèles

D'après les benchmarks disponibles publiquement, **Laya en zero-shot (poids
tels que téléchargés) a un accuracy plus faible qu'un baseline "toujours
répondre la classe majoritaire"** sur son propre benchmark — il faut le
fine-tuner sur ton domaine pour qu'il devienne réellement utile (voir le
notebook Kaggle du repo `NandhaKishorM/laya`). Jev fait mieux mais reste, sur
son propre benchmark, en dessous de LLMs généralistes utilisés simplement en
zero-shot. L'argument de vente des deux est la **latence/coût**, pas une
meilleure qualité de décision — ne fais pas confiance aux chiffres par défaut
sans les valider toi-même sur tes propres cas.

## Ressources

- Modèle : https://huggingface.co/convaiinnovations/laya
- Démo officielle (Space) : https://huggingface.co/spaces/convaiinnovations/laya-demo
- Package : https://pypi.org/project/laya/
