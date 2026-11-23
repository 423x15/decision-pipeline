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
    "Support Ticket Triage": {
        "description": "Reproduces the official model example: routing a customer email.",
        "state": (
            "From: client@acme-corp.com\n"
            "Subject: Double charge on invoice #4411\n"
            "Message: Hello, we were charged twice for the month of March. "
            "This is the second time this has happened and it is very frustrating. "
            "If this is not corrected quickly we will switch to a competitor."
        ),
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which department should handle this request?",
                "criteria": {
                    "billing": "Billing, payments, refunds",
                    "technical": "Bugs, technical issues",
                    "sales": "New sales, upsell",
                },
            },
            "urgency": {
                "type": "score",
                "instructions": "What is the urgency level of the request?",
                "criteria": ["not urgent", "address soon", "critical"],
            },
            "churn_risk": {
                "type": "noul",
                "instructions": "Is the customer threatening to leave for a competitor?",
            },
        },
    },
    "Content Moderation": {
        "description": "Detects the type and severity of a potentially problematic message.",
        "state": (
            "Comment posted on the community forum: "
            "\"Seriously, stop spamming this topic with your garbage links. "
            "You're only here to sell your stuff, this is a complete joke.\""
        ),
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "What is the main category of this message?",
                "criteria": {
                    "safe": "Acceptable content, normal tone",
                    "spam": "Unsolicited advertising or promotion",
                    "harassment": "Insult or attack directed at a person",
                    "hate": "Hate speech",
                },
            },
            "severity": {
                "type": "score",
                "instructions": "What is the severity of the content?",
                "criteria": ["benign", "moderate", "severe"],
            },
            "needs_human_review": {
                "type": "noul",
                "instructions": "Does this message require human review before action?",
            },
        },
    },
    "LLM Model Routing": {
        "description": "Decides which LLM tier (small/medium/large) should handle a user query.",
        "state": (
            "User query: \"Can you explain step by step how to prove that "
            "the sum of the first n odd integers equals n^2, then write "
            "a formal proof in Lean 4?\""
        ),
        "questions": {
            "best_model_tier": {
                "type": "choice",
                "instructions": "Which model tier is most appropriate for this query?",
                "criteria": {
                    "small": "Small/fast model, simple tasks",
                    "medium": "Intermediate model, moderate reasoning",
                    "large": "Most powerful model, complex reasoning",
                },
            },
            "complexity": {
                "type": "score",
                "instructions": "What level of reasoning complexity is required?",
                "criteria": ["simple", "moderate", "high"],
            },
            "needs_tools": {
                "type": "noul",
                "instructions": "Does this query require calling an external tool (code, search, calculation)?",
            },
        },
    },
    "Custom": {
        "description": "Start from a blank state and build your own typed questions.",
        "state": (
            "Paste your own text, email, ticket or JSON here to be evaluated by Laya."
        ),
        "questions": {
            "example_choice": {
                "type": "choice",
                "instructions": "Example choice question: edit me",
                "criteria": {"option_a": "Description A", "option_b": "Description B"},
            },
        },
    },
}

SCENARIO_ORDER = list(SCENARIOS.keys())
