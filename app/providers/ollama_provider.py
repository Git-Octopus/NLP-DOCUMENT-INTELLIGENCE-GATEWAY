"""
Ollama Model Provider for NLP Document Intelligence Gateway.

Provides integration with local Ollama instances for fallback topic resolution,
entity extraction, summarization, and sentiment analysis without external cloud dependencies.
"""

import json
import re
import logging
import requests
from typing import List, Dict, Any, Optional

from app.config import settings
from app.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """
    Ollama LLM provider using the local Ollama HTTP API.
    Model name is dynamic and never hardcoded.
    """

    def __init__(
        self, model_name: Optional[str] = None, base_url: Optional[str] = None
    ):
        from app.services.model_manager import model_manager

        chosen_model = model_name
        if not chosen_model:
            if getattr(model_manager, "reasoning_provider", None) == "ollama":
                chosen_model = getattr(model_manager, "reasoning_model", None)
            if not chosen_model or "gemini" in str(chosen_model).lower():
                installed_reasoning = (
                    model_manager.get_catalog()
                    .get("reasoning_catalog", {})
                    .get("ollama", [])
                )
                chosen_model = (
                    installed_reasoning[0]
                    if installed_reasoning
                    else getattr(settings, "OLLAMA_MODEL", "qwen2.5:7b")
                )

        self._model_name = chosen_model
        active_url = model_manager.get_ollama_url() or settings.OLLAMA_BASE_URL
        self._base_url = (base_url or active_url).rstrip("/")

    @property
    def provider_name(self) -> str:
        return "ollama"

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self._base_url}/api/tags", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    def _generate_text(self, prompt: str, as_json: bool = True) -> Optional[str]:
        """Sends generation prompt to Ollama /api/generate endpoint."""
        url = f"{self._base_url}/api/generate"
        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "stream": False,
        }
        if as_json:
            payload["format"] = "json"

        try:
            resp = requests.post(url, json=payload, timeout=20.0)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("response", "").strip()
            else:
                logger.warning(
                    f"Ollama returned HTTP {resp.status_code} for model {self._model_name}: {resp.text}"
                )
                return None
        except Exception as e:
            logger.warning(
                f"Ollama call failed ({e}). Falling back to heuristic resolution."
            )
            return None

    def _parse_json_from_text(
        self, response_text: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        if not response_text:
            return None
        cleaned = re.sub(r"^```json\s*", "", response_text.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"^```\s*", "", cleaned, flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except Exception:
            match = re.search(r"(\{.*\})", response_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except Exception:
                    pass
        return None

    def resolve_ambiguous(
        self,
        text: str,
        candidate_topics: List[Dict[str, Any]],
        available_taxonomy: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Resolves ambiguous document topics using Ollama model."""
        taxonomy_summary = ""
        if available_taxonomy:
            taxonomy_summary = "Available Topics and IDs:\n" + "\n".join(
                f"- {item.get('name')} (id: {item.get('id')}): {item.get('description')}"
                for item in available_taxonomy
            )

        candidates_str = (
            ", ".join(
                [
                    f"{c.get('topic')} (score: {c.get('score', 0):.2f})"
                    for c in candidate_topics
                ]
            )
            if candidate_topics
            else "None provided"
        )

        prompt = f"""You are an expert NLP classifier for customer support documents.
Classify the following customer text into the most appropriate topic and intent.

Document: "{text}"
Candidate Topics: {candidates_str}
{taxonomy_summary}

Respond ONLY with valid JSON in this exact structure:
{{
    "topic": "Name of best topic",
    "topic_id": "topic_id_in_snake_case",
    "intent": "specific_customer_intent",
    "confidence": float between 0.0 and 1.0,
    "reasoning": "brief explanation"
}}
"""
        response_text = self._generate_text(prompt, as_json=True)
        parsed = self._parse_json_from_text(response_text)
        if parsed and "topic" in parsed:
            return {
                "topic": str(parsed["topic"]),
                "topic_id": str(
                    parsed.get("topic_id", parsed["topic"].lower().replace(" ", "_"))
                ),
                "intent": str(parsed.get("intent", "general_inquiry")),
                "confidence": max(0.5, min(1.0, float(parsed.get("confidence", 0.75)))),
                "reasoning": str(parsed.get("reasoning", "Resolved by Ollama.")),
            }

        return self._heuristic_resolve(text, candidate_topics)

    def extract_domain_entities(
        self, text: str, existing_entities: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Extracts domain-specific entities using Ollama."""
        prompt = f"""Extract domain entities (e.g. PAYMENT_METHOD, PLAN_TIER, SOFTWARE_ERROR) from this text:
"{text}"

Return JSON: [{{"text": "exact substring", "label": "LABEL"}}]
"""
        response_text = self._generate_text(prompt, as_json=True)
        parsed = None
        if response_text:
            try:
                cleaned = re.sub(
                    r"^```json\s*", "", response_text.strip(), flags=re.MULTILINE
                )
                cleaned = re.sub(r"^```\s*", "", cleaned, flags=re.MULTILINE).strip()
                parsed = json.loads(cleaned)
            except Exception:
                match = re.search(r"(\[.*\])", response_text, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group(1))
                    except Exception:
                        pass

        results = []
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "text" in item and "label" in item:
                    entity_str = str(item["text"]).strip()
                    start_char = text.find(entity_str) if entity_str in text else None
                    end_char = (
                        start_char + len(entity_str)
                        if start_char is not None and start_char >= 0
                        else None
                    )
                    results.append(
                        {
                            "text": entity_str,
                            "entity": entity_str,
                            "label": str(item["label"]).upper(),
                            "source": "ollama",
                            "start_char": start_char,
                            "end_char": end_char,
                        }
                    )
            return results

        return self._heuristic_domain_entities(text)

    def summarize_text(
        self, text: str, max_length: Optional[int] = 150
    ) -> Dict[str, Any]:
        """Generates executive summary and bullet points using Ollama."""
        prompt = (
            f"Summarize this support document into a paragraph under {max_length or 150} "
            "words with 2-4 key bullet points.\n"
            f'Document: "{text}"\n'
            'Return JSON: {"summary": "...", "key_points": ["..."]}\n'
        )
        response_text = self._generate_text(prompt, as_json=True)
        parsed = self._parse_json_from_text(response_text) if response_text else None
        if parsed and "summary" in parsed and "key_points" in parsed:
            return {
                "summary": str(parsed["summary"]),
                "key_points": [str(p) for p in parsed.get("key_points", [])],
            }
        return self._heuristic_summarize(text)

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyzes sentiment using Ollama."""
        prompt = (
            f'Analyze sentiment polarity of: "{text}". '
            'Return JSON: {"sentiment": "positive"|"neutral"|"negative", "score": float from -1.0 to 1.0}'
        )
        response_text = self._generate_text(prompt, as_json=True)
        parsed = self._parse_json_from_text(response_text) if response_text else None
        if parsed and "sentiment" in parsed and "score" in parsed:
            return {
                "sentiment": str(parsed["sentiment"]).lower(),
                "score": max(-1.0, min(1.0, float(parsed["score"]))),
            }
        return self._heuristic_sentiment(text)

    def _heuristic_resolve(
        self, text: str, candidate_topics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if candidate_topics:
            best = candidate_topics[0]
            return {
                "topic": best.get("topic", "General Support"),
                "topic_id": best.get("topic_id", "general_support"),
                "intent": f"inquire_{best.get('topic_id', 'general_support')}",
                "confidence": max(0.65, float(best.get("score", 0.65))),
                "reasoning": f"Identified via semantic vector matching ({best.get('topic')}).",
            }
        return {
            "topic": "General Support",
            "topic_id": "general_support",
            "intent": "general_inquiry",
            "confidence": 0.70,
            "reasoning": "Heuristic fallback assignment.",
        }

    def _heuristic_domain_entities(self, text: str) -> List[Dict[str, Any]]:
        entities = []
        lower = text.lower()
        for pm in [
            "credit card",
            "debit card",
            "visa",
            "mastercard",
            "paypal",
            "apple pay",
            "upi",
        ]:
            if pm in lower:
                idx = lower.find(pm)
                entities.append(
                    {
                        "text": text[idx : idx + len(pm)],
                        "entity": text[idx : idx + len(pm)],
                        "label": "PAYMENT_METHOD",
                        "source": "ollama_fallback",
                        "start_char": idx,
                        "end_char": idx + len(pm),
                    }
                )
        return entities

    def _heuristic_summarize(self, text: str) -> Dict[str, Any]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        summary = " ".join(sentences[:2]) if sentences else text[:150]
        key_points = (
            [s for s in sentences[:3]] if sentences else ["Customer support inquiry."]
        )
        return {"summary": summary, "key_points": key_points}

    def _heuristic_sentiment(self, text: str) -> Dict[str, Any]:
        from app.nlp.sentiment import analyze_sentiment_local

        return analyze_sentiment_local(text)
