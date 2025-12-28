"""
LLM Extractor Module - Backward Compatibility Layer

This module re-exports from the new extraction/ and models/ packages
to maintain backward compatibility with existing code.
"""

# Re-export models for backward compatibility
from app.models.components import ChemicalComponent, MaterialComposition
from app.models.relationships import (
    ComponentRelationship,
    SubstitutionRelationship,
    SynergyRelationship,
)
from app.models.extraction import ExtractionResult

# Re-export from new extraction module
from app.extraction.extractors import LLMExtractor
from app.extraction.demo_data import get_demo_extraction_result

__all__ = [
    # Models
    "ChemicalComponent",
    "MaterialComposition",
    "ComponentRelationship",
    "SubstitutionRelationship",
    "SynergyRelationship",
    "ExtractionResult",
    # Extractor
    "LLMExtractor",
    "get_demo_extraction_result",
]
