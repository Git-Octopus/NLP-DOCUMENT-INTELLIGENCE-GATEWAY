import re
import math
from typing import List
from collections import Counter
from fastapi import APIRouter, status, HTTPException
from app.schemas.summarize import SummarizeRequest, SummarizeResponse
from app.config import settings
from app.providers import get_llm_provider

router = APIRouter(tags=["Summarization"])


def _extractive_summarize(text: str, max_words: int = 150) -> tuple[str, List[str]]:
    """
    Extractive summarization using sentence-level salience scoring based on term frequencies.
    Produces a coherent summary paragraph and salient bullet points.
    """
    raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not raw_sentences:
        raw_sentences = [text.strip()]

    total_words = len(text.split())
    if len(raw_sentences) <= 2 or total_words <= max_words:
        summary = text.strip()
        key_points = [s for s in raw_sentences if len(s.split()) >= 3][:3]
        if not key_points:
            key_points = [summary]
        return summary, key_points

    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    stop_words = {
        "the",
        "and",
        "is",
        "in",
        "to",
        "of",
        "it",
        "that",
        "you",
        "for",
        "on",
        "with",
        "as",
        "this",
        "was",
        "at",
        "by",
        "an",
        "be",
        "from",
        "or",
        "are",
        "have",
        "has",
        "had",
        "not",
        "but",
        "what",
        "all",
        "were",
        "when",
        "we",
        "there",
        "can",
        "your",
    }
    content_words = [w for w in words if w not in stop_words]
    word_freqs = Counter(content_words)
    max_freq = max(word_freqs.values()) if word_freqs else 1

    normalized_freqs = {w: count / max_freq for w, count in word_freqs.items()}

    sentence_scores = []
    for idx, sent in enumerate(raw_sentences):
        sent_words = re.findall(r"\b[a-zA-Z]{3,}\b", sent.lower())
        if not sent_words:
            continue
        score = sum(normalized_freqs.get(w, 0.0) for w in sent_words)
        position_weight = (
            1.25 if idx == 0 else (1.1 if idx == len(raw_sentences) - 1 else 1.0)
        )
        norm_score = (score / math.sqrt(len(sent_words))) * position_weight
        sentence_scores.append((idx, sent, norm_score))

    if not sentence_scores:
        return text.strip(), [text.strip()]

    ranked_by_score = sorted(sentence_scores, key=lambda x: x[2], reverse=True)

    selected_indices = set()
    current_word_count = 0

    for idx, sent, _ in ranked_by_score:
        sent_word_count = len(sent.split())
        if current_word_count + sent_word_count <= max_words or not selected_indices:
            selected_indices.add(idx)
            current_word_count += sent_word_count
        if current_word_count >= max_words:
            break

    summary_sentences = [raw_sentences[i] for i in sorted(selected_indices)]
    summary = " ".join(summary_sentences)
    key_points = [item[1] for item in ranked_by_score[:3]]

    return summary, key_points


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    status_code=status.HTTP_200_OK,
    summary="Summarize Support Text",
    description=(
        "Generates an executive summary and key highlight points using LLM providers "
        "or local extractive sentence ranking."
    ),
)
async def summarize_text(request: SummarizeRequest) -> SummarizeResponse:
    """
    Summarizes document content into a concise paragraph with key bullet points.
    Leverages external LLM if configured; otherwise runs high-quality local extractive summarization.
    """
    cleaned_text = request.text.strip()
    if len(cleaned_text) < 10:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Text input for summarization must be at least 10 characters long.",
        )

    original_word_count = len(cleaned_text.split())
    target_max_words = request.max_length or 150

    summary = ""
    key_points = []

    # 1. Try LLM Provider if configured
    if settings.OPENAI_API_KEY or settings.GEMINI_API_KEY:
        try:
            provider = get_llm_provider()
            llm_result = provider.summarize_text(
                cleaned_text, max_length=target_max_words
            )
            if llm_result and llm_result.get("summary"):
                summary = llm_result["summary"]
                key_points = llm_result.get("key_points", [])
        except Exception:
            pass

    # 2. Local Extractive Summarization Fallback
    if not summary:
        summary, key_points = _extractive_summarize(
            cleaned_text, max_words=target_max_words
        )

    return SummarizeResponse(
        summary=summary,
        key_points=key_points,
        original_word_count=original_word_count,
        summary_word_count=len(summary.split()),
    )
