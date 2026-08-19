"""
Named Entity Recognition (NER) module for NLP Document Intelligence Gateway.

Combines spaCy statistical NER models (en_core_web_sm / md / lg) with
rule-based regex extractors to produce a unified entity list.
"""

from typing import List, Dict, Any

def extract_spacy_entities(text: str) -> List[Dict[str, Any]]:
    """
    Extracts entities using spaCy's pretrained NER pipeline:
    - PERSON, ORG, GPE, PRODUCT, EVENT, LAW, etc.
    
    Returns list of dicts matching ExtractedEntity schema attributes:
    {'text': str, 'label': str, 'source': 'spacy', 'start_char': int, 'end_char': int}
    """
    pass

def extract_all_entities(text: str) -> List[Dict[str, Any]]:
    """
    Combines rule-based regex entities with spaCy NER entities.
    Applies deduplication and resolving of character index overlaps.
    Matches 'entities' field in AnalyzeResponse schema.
    """
    pass
