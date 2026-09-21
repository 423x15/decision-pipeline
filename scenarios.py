from __future__ import annotations

from typing import Any

# Each scenario provides a "state" (what Laya evaluates) and "questions"
# in the exact shape expected by agent.predict(state, questions):
#   { key: {"type": "choice"|"score"|"noul", "instructions": str, "criteria": ...} }
#
# criteria:
#   - choice -> dict[label, description]
#   - score  -> list[str] ordered from low to high
#   - noul   -> omitted (yes/no-ish calibrated probability question)

SCENARIOS: dict[str, dict[str, Any]] = {
    "Triage ticket support": {
        "description": "Reprend l'exemple officiel du modèle : router un email client.",
        "state": (
            "De: client@acme-corp.com\n"
            "Objet: Facturation en double sur la facture #4411\n"
            "Message: Bonjour, nous avons été prélevés deux fois pour le mois de mars. "
            "C'est la deuxième fois que ça arrive et c'est très frustrant, si ce n'est "
            "pas corrigé rapidement nous allons changer de prestataire."
        ),
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Quel service doit traiter cette demande ?",
                "criteria": {
                    "billing": "Facturation, paiements, remboursements",
                    "technical": "Bugs, problèmes techniques",
                    "sales": "Nouvelles ventes, upsell",
                },
            },
            "urgency": {
                "type": "score",
                "instructions": "Quel est le niveau d'urgence de la demande ?",
                "criteria": ["pas urgent", "à traiter bientôt", "critique"],
            },
            "churn_risk": {
                "type": "noul",
                "instructions": "Le client menace-t-il de partir chez un concurrent ?",
            },
        },
    },
    "Modération de contenu": {
        "description": "Détecte le type et la sévérité d'un message potentiellement problématique.",
        "state": (
            "Commentaire posté sur le forum communautaire : "
            "\"Sérieusement, arrêtez de spammer ce topic avec vos liens pourris, "
            "vous êtes juste ici pour vendre votre truc, c'est du foutage de gueule.\""
        ),
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "Quelle est la catégorie principale du message ?",
                "criteria": {
                    "safe": "Contenu acceptable, ton normal",
                    "spam": "Publicité ou promotion non sollicitée",
                    "harassment": "Insulte ou attaque envers une personne",
                    "hate": "Discours de haine",
                },
            },
            "severity": {
                "type": "score",
                "instructions": "Quelle est la sévérité du contenu ?",
                "criteria": ["bénin", "modéré", "sévère"],
            },
            "needs_human_review": {
                "type": "noul",
                "instructions": "Ce message nécessite-t-il une revue humaine avant action ?",
            },
        },
    },
    "Routage de modèle LLM": {
        "description": "Décide quel modèle LLM (petit/moyen/gros) doit traiter une requête utilisateur.",
        "state": (
            "Requête utilisateur : \"Peux-tu m'expliquer étape par étape comment "
            "prouver que la somme des n premiers entiers impairs vaut n^2, puis "
            "écrire une preuve formelle en Lean 4 ?\""
        ),
        "questions": {
            "best_model_tier": {
                "type": "choice",
                "instructions": "Quel niveau de modèle est le plus adapté pour traiter cette requête ?",
                "criteria": {
                    "small": "Modèle petit/rapide, tâches simples",
                    "medium": "Modèle intermédiaire, raisonnement modéré",
                    "large": "Modèle le plus puissant, raisonnement complexe",
                },
            },
            "complexity": {
                "type": "score",
                "instructions": "Quelle est la complexité de raisonnement requise ?",
                "criteria": ["simple", "modérée", "élevée"],
            },
            "needs_tools": {
                "type": "noul",
                "instructions": "Cette requête nécessite-t-elle d'appeler un outil externe (code, recherche, calcul) ?",
            },
        },
    },
    "Personnalisé": {
        "description": "Partez d'un état vide et construisez vos propres questions typées.",
        "state": (
            "Collez ici votre propre texte, email, ticket ou JSON à faire évaluer par Laya."
        ),
        "questions": {
            "example_choice": {
                "type": "choice",
                "instructions": "Exemple de question à choix : modifiez-moi",
                "criteria": {"option_a": "Description A", "option_b": "Description B"},
            },
        },
    },
}

SCENARIO_ORDER = list(SCENARIOS.keys())
