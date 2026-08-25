"""
Named Entity Recognition (NER) module for NLP Document Intelligence Gateway.

Combines spaCy statistical NER models with rule-based regex extractors.
Rule-based entities strictly take priority in case of overlapping character spans.
"""

from typing import List, Dict, Any
from app.nlp.linguistic import get_spacy_model
from app.nlp.rules import extract_rule_entities

def extract_spacy_entities(text: str) -> List[Dict[str, Any]]:
    """
    Runs spaCy pretrained NER to detect PERSON, ORG, GPE, DATE, MONEY, CARDINAL, PRODUCT, etc.
    """
    if not text or not text.strip():
        return []
        
    try:
        nlp = get_spacy_model()
        doc = nlp(text)
        
        entities = []
        for ent in doc.ents:
            ent_text = ent.text.strip()
            if not ent_text:
                continue
            entities.append({
                "entity": ent_text,
                "text": ent_text,           # schema compatibility
                "label": ent.label_,
                "source": "spacy",
                "start": ent.start_char,
                "end": ent.end_char,
                "start_char": ent.start_char, # schema compatibility
                "end_char": ent.end_char      # schema compatibility
            })
        return entities
    except Exception:
        # Fallback if spaCy is unavailable
        return []

def extract_all_entities(text: str) -> List[Dict[str, Any]]:
    """
    Combines rule-based entities with spaCy NER entities.
    Rule-based entities MUST take priority over spaCy entities in case of overlapping character spans.
    Returns deduplicated, non-overlapping entities sorted by character position.
    """
    if not text or not text.strip():
        return []
        
    rule_entities = extract_rule_entities(text)
    spacy_entities = extract_spacy_entities(text)
    
    selected_entities = []
    occupied_spans = []
    
    # 1. First add all rule-based entities (High Priority)
    for r_ent in rule_entities:
        selected_entities.append(r_ent)
        occupied_spans.append((r_ent["start"], r_ent["end"]))
        
    # 2. Add spaCy entities only if they do NOT overlap with any existing entity span
    for s_ent in spacy_entities:
        s_start, s_end = s_ent["start"], s_ent["end"]
        
        # Check for character range overlap: max(s1, s2) < min(e1, e2)
        has_overlap = any(
            max(s_start, o_start) < min(s_end, o_end)
            for o_start, o_end in occupied_spans
        )
        
        if not has_overlap:
            selected_entities.append(s_ent)
            occupied_spans.append((s_start, s_end))
            
    # Sort final entities by start character index
    selected_entities.sort(key=lambda x: x["start"])
    return selected_entities
