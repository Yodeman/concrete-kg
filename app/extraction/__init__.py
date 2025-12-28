"""Extraction package for LLM-based knowledge extraction."""

from app.extraction.extractors import LLMExtractor
from app.extraction.demo_data import get_demo_extraction_result
from app.extraction.prompts import (
    COMPOSITION_EXTRACTION_PROMPT,
    SUBSTITUTION_EXTRACTION_PROMPT,
    SYNERGY_EXTRACTION_PROMPT,
)

__all__ = [
    "LLMExtractor",
    "get_demo_extraction_result",
    "COMPOSITION_EXTRACTION_PROMPT",
    "SUBSTITUTION_EXTRACTION_PROMPT",
    "SYNERGY_EXTRACTION_PROMPT",
]
