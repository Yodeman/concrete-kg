"""Relationship type definitions for the knowledge graph."""

from typing import Dict, Optional, Literal
from dataclasses import dataclass


@dataclass
class RelationshipTypeConfig:
    """Configuration for a relationship type."""
    color: str
    style: Literal["solid", "dashed", "dotted"]
    directed: bool
    description: str
    category: str  # "composition", "effect", "discovery"


# All relationship types with their visual and semantic properties
RELATIONSHIP_TYPES: Dict[str, RelationshipTypeConfig] = {
    # Composition relationships
    "CONTAINS": RelationshipTypeConfig(
        color="#10B981",  # Green
        style="solid",
        directed=True,
        description="Material contains a component",
        category="composition"
    ),
    "CITED_IN": RelationshipTypeConfig(
        color="#9C27B0",  # Purple
        style="dashed",
        directed=True,
        description="Referenced in source document",
        category="composition"
    ),
    
    # Effect relationships
    "AFFECTS": RelationshipTypeConfig(
        color="#FF9800",  # Orange
        style="solid",
        directed=True,
        description="Component affects a property",
        category="effect"
    ),
    "REACTS_WITH": RelationshipTypeConfig(
        color="#F44336",  # Red
        style="solid",
        directed=False,
        description="Components react chemically",
        category="effect"
    ),
    "PRODUCES": RelationshipTypeConfig(
        color="#00BCD4",  # Cyan
        style="solid",
        directed=True,
        description="Reaction produces a compound",
        category="effect"
    ),
    
    # Discovery relationships (NEW)
    "SUBSTITUTES": RelationshipTypeConfig(
        color="#3B82F6",  # Blue
        style="dashed",
        directed=True,
        description="Can substitute for another component",
        category="discovery"
    ),
    "SYNERGY_WITH": RelationshipTypeConfig(
        color="#22C55E",  # Green
        style="solid",
        directed=False,
        description="Positive synergistic effect when combined",
        category="discovery"
    ),
    "ANTAGONISTIC_TO": RelationshipTypeConfig(
        color="#EF4444",  # Red
        style="solid",
        directed=False,
        description="Negative antagonistic effect when combined",
        category="discovery"
    ),
    "SIMILAR_TO": RelationshipTypeConfig(
        color="#A855F7",  # Purple
        style="dotted",
        directed=False,
        description="Materials with similar properties/composition",
        category="discovery"
    ),
    
    # Experiment relationships (NEW)
    "VALIDATES": RelationshipTypeConfig(
        color="#EC4899",  # Pink
        style="solid",
        directed=True,
        description="Experiment validates material properties",
        category="experiment"
    ),
    "CONDITIONS": RelationshipTypeConfig(
        color="#06B6D4",  # Cyan
        style="solid",
        directed=True,
        description="Experiment performed under conditions",
        category="experiment"
    ),
}


def get_relationship_color(rel_type: str) -> str:
    """
    Get the display color for a relationship type.
    
    Args:
        rel_type: The relationship type name
        
    Returns:
        Hex color string
    """
    if rel_type in RELATIONSHIP_TYPES:
        return RELATIONSHIP_TYPES[rel_type].color
    return "#9CA3AF"  # Default gray


def get_relationship_style(rel_type: str) -> str:
    """
    Get the line style for a relationship type.
    
    Args:
        rel_type: The relationship type name
        
    Returns:
        Style string: 'solid', 'dashed', or 'dotted'
    """
    if rel_type in RELATIONSHIP_TYPES:
        return RELATIONSHIP_TYPES[rel_type].style
    return "solid"


def is_directed(rel_type: str) -> bool:
    """Check if a relationship type is directed."""
    if rel_type in RELATIONSHIP_TYPES:
        return RELATIONSHIP_TYPES[rel_type].directed
    return True


def get_relationship_types() -> list:
    """Get list of all relationship type names."""
    return list(RELATIONSHIP_TYPES.keys())


def get_relationship_types_by_category(category: str) -> list:
    """Get relationship types filtered by category."""
    return [
        name for name, config in RELATIONSHIP_TYPES.items()
        if config.category == category
    ]


def get_discovery_relationship_types() -> list:
    """Get relationship types used for material discovery."""
    return get_relationship_types_by_category("discovery")
