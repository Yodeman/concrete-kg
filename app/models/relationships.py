"""Relationship models for knowledge graph edges."""

from typing import Literal, Optional
from pydantic import Field

from app.models.base import LLMExportableModel


class ComponentRelationship(LLMExportableModel):
    """
    Represents a general relationship between components in concrete.
    
    This is the base relationship type for capturing interactions
    and dependencies between materials and their components.
    """
    
    source: str = Field(
        ...,
        description="Source component name or formula"
    )
    target: str = Field(
        ...,
        description="Target component name or formula"
    )
    relationship_type: str = Field(
        ...,
        description="Type: 'CONTAINS', 'AFFECTS', 'REACTS_WITH', 'PRODUCES', 'CITED_IN'"
    )
    description: str = Field(
        ...,
        description="Description of the relationship"
    )
    strength: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Relationship strength/importance (0-1 scale)"
    )


class SubstitutionRelationship(LLMExportableModel):
    """
    Represents a material substitution relationship.
    
    Captures when one material can partially or fully replace another
    in a concrete mixture, including the effects and conditions.
    """
    
    original: str = Field(
        ...,
        description="Original component being replaced (e.g., 'Portland Cement')"
    )
    substitute: str = Field(
        ...,
        description="Replacement component (e.g., 'Fly Ash')"
    )
    max_ratio: float = Field(
        ...,
        ge=0,
        le=1,
        description="Maximum substitution ratio (0-1, e.g., 0.3 = 30% replacement)"
    )
    effects: str = Field(
        ...,
        description="Effect on properties (e.g., 'Reduced early strength, improved durability')"
    )
    conditions: Optional[str] = Field(
        default=None,
        description="Conditions when substitution is valid (e.g., 'Class F fly ash only')"
    )
    property_changes: Optional[dict] = Field(
        default=None,
        description="Quantitative property changes (e.g., {'strength_7d': -15, 'durability': +20})"
    )


class SynergyRelationship(LLMExportableModel):
    """
    Represents synergistic or antagonistic effects between components.
    
    Captures when combining two components produces effects greater
    than the sum of individual effects (synergy) or negative interactions.
    """
    
    component1: str = Field(
        ...,
        description="First component name or formula"
    )
    component2: str = Field(
        ...,
        description="Second component name or formula"
    )
    effect_type: Literal["SYNERGY", "ANTAGONISTIC"] = Field(
        ...,
        description="Type of interaction: 'SYNERGY' (positive) or 'ANTAGONISTIC' (negative)"
    )
    effect: str = Field(
        ...,
        description="Description of the combined effect"
    )
    strength: float = Field(
        ...,
        ge=0,
        le=1,
        description="Effect strength (0-1 scale, higher = stronger effect)"
    )
    mechanism: Optional[str] = Field(
        default=None,
        description="Mechanism explaining the interaction"
    )
    conditions: Optional[str] = Field(
        default=None,
        description="Conditions when this effect occurs"
    )


class SimilarityRelationship(LLMExportableModel):
    """
    Represents similarity between materials.
    
    Used for material discovery by identifying materials with
    comparable compositions or properties.
    """
    
    material1: str = Field(
        ...,
        description="First material name"
    )
    material2: str = Field(
        ...,
        description="Second material name"
    )
    similarity_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Similarity score (0-1, higher = more similar)"
    )
    similarity_basis: str = Field(
        ...,
        description="Basis for similarity: 'composition', 'properties', 'application'"
    )
    shared_components: Optional[list] = Field(
        default=None,
        description="List of shared major components"
    )
