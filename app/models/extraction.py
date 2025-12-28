"""Extraction result aggregate model."""

from typing import List, Optional
from pydantic import Field

from app.models.base import LLMExportableModel
from app.models.components import MaterialComposition
from app.models.relationships import (
    ComponentRelationship,
    SubstitutionRelationship,
    SynergyRelationship,
)
from app.models.properties import QuantitativeProperty


class ExtractionResult(LLMExportableModel):
    """
    Complete extraction result from a document.
    
    Aggregates all extracted knowledge including materials,
    relationships, substitutions, and synergies.
    """
    
    # Core extractions
    materials: List[MaterialComposition] = Field(
        default_factory=list,
        description="List of extracted material compositions"
    )
    relationships: List[ComponentRelationship] = Field(
        default_factory=list,
        description="General relationships between components"
    )
    
    # Enhanced extractions for material discovery
    substitutions: List[SubstitutionRelationship] = Field(
        default_factory=list,
        description="Material substitution relationships"
    )
    synergies: List[SynergyRelationship] = Field(
        default_factory=list,
        description="Synergistic and antagonistic effects"
    )
    quantitative_properties: List[QuantitativeProperty] = Field(
        default_factory=list,
        description="Quantitative property measurements"
    )
    
    # Metadata
    source_document: Optional[str] = Field(
        default=None,
        description="Source document title or identifier"
    )
    extraction_confidence: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Overall confidence in extraction quality (0-1)"
    )
    
    def get_summary(self) -> str:
        """Get a summary of the extraction result."""
        return (
            f"Extracted {len(self.materials)} materials, "
            f"{len(self.relationships)} relationships, "
            f"{len(self.substitutions)} substitutions, "
            f"{len(self.synergies)} synergies"
        )
    
    def has_discovery_data(self) -> bool:
        """Check if extraction contains discovery-relevant data."""
        return len(self.substitutions) > 0 or len(self.synergies) > 0
