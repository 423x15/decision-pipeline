# Laya Playground

Petit dashboard Streamlit pour explorer [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) :
un moteur de décision **non-autorégressif** de 421M paramètres, qui prend un
`state` (texte, email, ticket, JSON) et des **questions typées** (`choice`,
`score`, `noul`) et répond en une seule passe (~33ms sur GPU) avec des
probabilités calibrées — sans génération de texte.

## Démarrage rapide (mode démo, sans rien télécharger)

```bash
cd laya_demo
python -m venv .venv && source .venv/bin/activate   # optionnel
pip install streamlit pandas
streamlit run app.py
```

Ouvre `http://localhost:8501`, laisse le mode **"Démo simulée"** sélectionné
dans la barre latérale, choisis un scénario (triage support, modération de
contenu, routage de modèle LLM...) et clique sur **Lancer la décision**.
Les réponses sont simulées (déterministes, basées sur le texte saisi) pour
que tu puisses te familiariser avec l'interface immédiatement.

## Utiliser le vrai modèle

```bash
pip install -r requirements.txt
```

Puis dans la barre latérale, passe sur **"Modèle réel (local)"** et clique sur
**Charger le modèle Laya**. Le premier chargement télécharge les poids depuis
Hugging Face (~421M paramètres) — il faut donc une connexion internet à ce
moment-là. Ensuite le modèle tourne 100% en local (CPU ou GPU si dispo).

Si le chargement semble bloqué, essaie de lancer avec `USE_TF=0` :

```bash
USE_TF=0 streamlit run app.py
```

## Ce que fait l'appli

- 4 scénarios prêts à l'emploi (triage support, modération, routage LLM) +
  un mode **Personnalisé** où tu édites toi-même les questions (clé, type
  `choice`/`score`/`noul`, instructions, critères) dans un tableau éditable.
- Visualisation dédiée par type de question :
  - `choice` → barres de probabilité par option, gagnant mis en valeur
  - `score` → jauge/curseur sur l'échelle ordinale définie
  - `noul` → jauge de probabilité calibrée (vert → jaune → rouge)
- Latence de la passe d'inférence affichée en gros (l'argument phare du
  modèle : ~33ms).
- Historique des runs dans la barre latérale.
- Réponse JSON brute disponible dans un expander pour debug.

## Note sur le format exact des réponses

Le champ exact renvoyé par `agent.predict(...)` (ex. `probabilities` vs
`scores` vs `distribution`) a été reconstitué à partir de la description du
package sur PyPI, faute d'accès direct à `huggingface.co` depuis cet
environnement au moment de la création du projet. Le rendu (`ui.py`) essaie
plusieurs noms de clés courants et affiche toujours le JSON brut en repli —
si un champ ne s'affiche pas correctement une fois le vrai modèle chargé,
regarde la structure dans l'expander JSON et ajuste `_get_first(...)` dans
`ui.py` en conséquence (c'est une seule ligne par type de question).

## Ressources

- Modèle : https://huggingface.co/convaiinnovations/laya
- Démo officielle (Space) : https://huggingface.co/spaces/convaiinnovations/laya-demo
- Package : https://pypi.org/project/laya/
