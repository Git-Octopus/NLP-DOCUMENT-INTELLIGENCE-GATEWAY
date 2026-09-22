"""
Google Gemini Model Provider for NLP Document Intelligence Gateway.
"""

import json
import re
import logging
from typing import List, Dict, Any, Optional

from app.config import settings
from app.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class GeminiProvider(BaseLLMProvider):
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self._api_key = api_key or settings.GEMINI_API_KEY
        self._model_name = model_name or settings.GEMINI_MODEL or "gemini-1.5-flash"
        self._client = None
        self._init_client()

    @property
    def provider_name(self) -> str:
        return "gemini"

    def is_available(self) -> bool:
        return bool(
            self._api_key and self._api_key.strip() and self._client is not None
        )

    def _init_client(self):
        if not self._api_key or not self._api_key.strip():
            logger.info(
                "Gemini API key not configured. Gemini provider running in fallback mock mode."
            )
            return
        try:
            import google.generativeai as genai

            genai.configure(api_key=self._api_key)
            self._client = genai.GenerativeModel(self._model_name)
            logger.info(
                f"Gemini GenerativeModel initialized successfully with model: {self._model_name}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to initialize Gemini client ({e}). Provider will use heuristics."
            )
            self._client = None

    def _generate_text(self, prompt: str) -> Optional[str]:
        if not self.is_available():
            return None
        try:
            response = self._client.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API request failed: {e}")
        return None

    def _parse_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        try:
            cleaned = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
            cleaned = re.sub(r"^```\s*", "", cleaned, flags=re.MULTILINE).strip()
            return json.loads(cleaned)
        except Exception as e:
            match = re.search(r"(\{.*\})", text, re.DOTALL)
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
        candidates_str = (
            ", ".join(
                [
                    f"{c.get('topic')} (score: {c.get('score', 0.0):.2f})"
                    for c in candidate_topics
                ]
            )
            or "None"
        )
        prompt = f"""Classify the customer support message into a single topic and intent.
Candidate topics: {candidates_str}
Customer Message: "{text}"
Return strictly JSON with keys: "topic", "topic_id", "intent", "confidence" (0.7-0.99), "reasoning".
"""
        response_text = self._generate_text(prompt)
        parsed = self._parse_json_from_text(response_text) if response_text else None
        if parsed and "topic" in parsed:
            return {
                "topic": parsed.get("topic", "General Support"),
                "topic_id": parsed.get("topic_id", "general_support"),
                "intent": parsed.get("intent", "general_inquiry"),
                "confidence": float(parsed.get("confidence", 0.85)),
                "reasoning": parsed.get(
                    "reasoning", "Resolved by Gemini fallback reasoning."
                ),
            }
        return self._heuristic_resolve(text, candidate_topics)

    def extract_domain_entities(
        self, text: str, existing_entities: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        existing_str = (
            ", ".join(
                [
                    f"{e.get('label')}: {e.get('text')}"
                    for e in (existing_entities or [])
                ]
            )
            or "None"
        )
        prompt = f"""Extract domain entities (PAYMENT_METHOD, PRODUCT_NAME, ERROR_CODE, TRACKING_NUMBER, SUBSCRIPTION_TIER, ACCOUNT_TYPE) not already extracted: {existing_str}.
Message: "{text}"
Return JSON list: [{{"text": "...", "label": "...", "source": "llm_fallback"}}]
"""
        response_text = self._generate_text(prompt)
        parsed = None
        if response_text:
            try:
                cleaned = re.sub(r"^```json\s*", "", response_text, flags=re.MULTILINE)
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
                            "source": "llm_fallback",
                            "start_char": start_char,
                            "end_char": end_char,
                        }
                    )
            return results

        return self._heuristic_domain_entities(text)

    def summarize_text(
        self, text: str, max_length: Optional[int] = 150
    ) -> Dict[str, Any]:
        prompt = f"""Summarize this support document into a paragraph under {max_length or 150} words with 2-4 key bullet points.
Document: "{text}"
Return JSON: {{"summary": "...", "key_points": ["..."]}}
"""
        response_text = self._generate_text(prompt)
        parsed = self._parse_json_from_text(response_text) if response_text else None
        if parsed and "summary" in parsed and "key_points" in parsed:
            return {
                "summary": str(parsed["summary"]),
                "key_points": [str(p) for p in parsed.get("key_points", [])],
            }
        return self._heuristic_summarize(text)

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        prompt = f"""Analyze sentiment polarity of: "{text}". Return JSON: {{"sentiment": "positive"|"neutral"|"negative", "score": float from -1.0 to 1.0}}"""
        response_text = self._generate_text(prompt)
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
                        "source": "llm_fallback",
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
