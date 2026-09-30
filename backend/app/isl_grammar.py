"""
Indian Sign Language (ISL) Linguistic Grammar & Context Interpretation Engine.
Linguistic Foundations:
- Syntactic Structure: Subject-Object-Verb (SOV) default word order, Topic-Comment framing.
- Temporal Framing: Time-first markers establish reference window without verb inflection.
- Interrogatives & Negation: Wh-question markers and negation morphemes occur at clause-final positions.
- Pragmatic Courtesy & Deixis: Spatial indexing (pronouns YOU, ME) and courtesy particles (PLEASE, THANK_YOU).
References:
- Zeshan, Ulrike (2000). "Sign Language in Indo-Pakistan: A Description of a Signed Language".
- Indian Sign Language Research and Training Centre (ISLRTC) Linguistic Grammar Guidelines.
"""

import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class GlossToken(BaseModel):
    """Represents a single recognized sign/gloss in the temporal stream."""
    gloss: str
    confidence: float = 1.0
    timestamp: float = Field(default_factory=time.time)
    duration_ms: float = 0.0
    is_uncertain: bool = False


class InterpretationResult(BaseModel):
    """Structured representation of an interpreted ISL sequence."""
    raw_glosses: List[str]
    gloss_sequence_str: str
    syntactic_structure: str
    interpreted_english: str
    grammatical_notes: str
    is_canonical_isl: bool
    confidence_score: float
    uncertainty_warning: Optional[str] = None


# Authoritative Grammar Patterns and Rule Mappings
CANONICAL_ISL_RULES = [
    {
        "pattern": ["HELLO"],
        "structure": "GREETING",
        "english": "Hello!",
        "notes": "Isolated formal greeting."
    },
    {
        "pattern": ["THANK_YOU"],
        "structure": "POLITENESS_MARKER",
        "english": "Thank you very much.",
        "notes": "Expressive gratitude marker."
    },
    {
        "pattern": ["PLEASE", "HELP"],
        "structure": "POLITENESS + PREDICATE",
        "english": "Please help me.",
        "notes": "Polite request; implicit first-person patient in ISL context."
    },
    {
        "pattern": ["HELP", "PLEASE"],
        "structure": "PREDICATE + POLITENESS",
        "english": "Please help me.",
        "notes": "Request with postposed politeness marker."
    },
    {
        "pattern": ["ME", "HELP"],
        "structure": "TOPIC(PATIENT) + ACTION",
        "english": "Help me.",
        "notes": "First-person patient followed by imperative action."
    },
    {
        "pattern": ["ME", "WATER"],
        "structure": "TOPIC + OBJECT_NEED",
        "english": "I want water.",
        "notes": "ISL Topic-Need construction: Subject/Topic followed by desired entity."
    },
    {
        "pattern": ["ME", "FOOD"],
        "structure": "TOPIC + OBJECT_NEED",
        "english": "I want food.",
        "notes": "Topic-Need construction: First-person followed by consumable entity."
    },
    {
        "pattern": ["ME", "FOOD", "WATER"],
        "structure": "TOPIC + OBJECT_LIST",
        "english": "I need food and water.",
        "notes": "Asyndetic coordination: Conjunctive words like 'and' are typically omitted in ISL."
    },
    {
        "pattern": ["YOU", "NAME"],
        "structure": "TOPIC + IDENTITY",
        "english": "What is your name?",
        "notes": "Elliptical Wh-interrogative in ISL: Topic + Identity noun implies 'What is your name?'."
    },
    {
        "pattern": ["YOU", "GOOD"],
        "structure": "TOPIC + EVALUATION",
        "english": "Are you doing good?",
        "notes": "Polar (yes/no) interrogative formed by Topic + Predicate adjective with non-manual tilt."
    },
    {
        "pattern": ["ME", "GOOD"],
        "structure": "TOPIC + EVALUATION",
        "english": "I am doing well.",
        "notes": "Declarative topic evaluation."
    },
    {
        "pattern": ["YES"],
        "structure": "AFFIRMATION",
        "english": "Yes, that is correct.",
        "notes": "Affirmative discourse marker."
    },
    {
        "pattern": ["NO"],
        "structure": "NEGATION",
        "english": "No, that is not correct.",
        "notes": "Negative discourse marker."
    },
    {
        "pattern": ["ME", "NO"],
        "structure": "TOPIC + NEGATION",
        "english": "I do not agree.",
        "notes": "Clause-final negation attached to first-person subject."
    },
    {
        "pattern": ["YOU", "HELP", "ME"],
        "structure": "AGENT + ACTION + PATIENT (SOV/SVO Context)",
        "english": "Can you please help me?",
        "notes": "Directional sign structure: Motion from addressee (YOU) toward signer (ME)."
    }
]


class ISLGrammarEngine:
    """Interprets raw gloss sequences into grammatically canonical natural language."""

    def __init__(self):
        self.rules = CANONICAL_ISL_RULES

    def interpret_sequence(self, tokens: List[GlossToken]) -> InterpretationResult:
        """
        Analyzes a sequence of temporal gloss tokens and produces a linguistic interpretation.
        """
        if not tokens:
            return InterpretationResult(
                raw_glosses=[],
                gloss_sequence_str="",
                syntactic_structure="EMPTY",
                interpreted_english="",
                grammatical_notes="No signs recognized.",
                is_canonical_isl=False,
                confidence_score=0.0,
                uncertainty_warning="Empty gloss sequence."
            )

        raw_glosses = [t.gloss.upper() for t in tokens if t.gloss]
        gloss_str = " ".join(raw_glosses)
        avg_conf = sum(t.confidence for t in tokens) / len(tokens) if tokens else 0.0

        # 1. Exact Pattern Match against documented ISL grammar corpus
        for rule in self.rules:
            if rule["pattern"] == raw_glosses:
                return InterpretationResult(
                    raw_glosses=raw_glosses,
                    gloss_sequence_str=gloss_str,
                    syntactic_structure=rule["structure"],
                    interpreted_english=rule["english"],
                    grammatical_notes=rule["notes"],
                    is_canonical_isl=True,
                    confidence_score=round(avg_conf, 3),
                    uncertainty_warning="Low recognition confidence across one or more signs in sequence." if avg_conf < 0.70 else None
                )

        # 2. Heuristic Syntactic Parsing for Multi-Sign Sequences
        # Identifies Topic, Objects, Verbs, and Clause-Final Markers
        has_question = "NAME" in raw_glosses or "WHAT" in raw_glosses
        has_negation = "NO" in raw_glosses or "NOT" in raw_glosses
        has_politeness = "PLEASE" in raw_glosses or "THANK_YOU" in raw_glosses

        components = []
        for g in raw_glosses:
            if g in ("ME", "YOU"):
                components.append("PRONOUN")
            elif g in ("WATER", "FOOD"):
                components.append("ENTITY")
            elif g in ("HELP", "EAT"):
                components.append("ACTION")
            elif g in ("GOOD",):
                components.append("DESCRIPTOR")
            elif g in ("PLEASE", "THANK_YOU"):
                components.append("COURTESY")
            elif g in ("YES", "NO"):
                components.append("POLARITY")
            else:
                components.append("LEXICAL")

        syntactic_str = " + ".join(components)

        # Construct rule-based linear interpretation
        words_mapped = []
        for g in raw_glosses:
            if g == "ME":
                words_mapped.append("I")
            elif g == "YOU":
                words_mapped.append("you")
            elif g == "HELP":
                words_mapped.append("help")
            elif g == "WATER":
                words_mapped.append("water")
            elif g == "FOOD":
                words_mapped.append("food")
            elif g == "GOOD":
                words_mapped.append("good")
            elif g == "PLEASE":
                words_mapped.append("please")
            elif g == "THANK_YOU":
                words_mapped.append("thank you")
            elif g == "HELLO":
                words_mapped.append("hello")
            elif g == "NAME":
                words_mapped.append("name")
            else:
                words_mapped.append(g.lower())

        approx_english = " ".join(words_mapped).capitalize()
        if not approx_english.endswith((".", "!", "?")):
            approx_english += "?" if has_question else "."

        uncertainty = None
        if avg_conf < 0.70:
            uncertainty = "Low recognition confidence across one or more signs in sequence."
        elif len(raw_glosses) > 1 and not any(rule["pattern"] == raw_glosses for rule in self.rules):
            uncertainty = "Uncatalogued ISL gloss combination. Verify word order with an ISL signer."

        return InterpretationResult(
            raw_glosses=raw_glosses,
            gloss_sequence_str=gloss_str,
            syntactic_structure=syntactic_str,
            interpreted_english=approx_english,
            grammatical_notes="Parsed using generalized ISL Topic-Comment heuristic.",
            is_canonical_isl=False,
            confidence_score=round(avg_conf, 3),
            uncertainty_warning=uncertainty
        )


isl_grammar_engine = ISLGrammarEngine()
