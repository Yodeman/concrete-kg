"""Node type definitions for the knowledge graph."""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class NodeTypeConfig:
    """Configuration for a node type."""
    color: str
    icon: str
    description: str


# Core node types with their visual properties
NODE_TYPES: Dict[str, NodeTypeConfig] = {
    # Existing types
    "Material": NodeTypeConfig(
        color="#10B981",  # Emerald green
        icon="🧱",
        description="Concrete materials and mixtures (cement, SCMs, aggregates)"
    ),
    "Component": NodeTypeConfig(
        color="#3B82F6",  # Blue
        icon="🧪",
        description="Chemical components and compounds (CaO, SiO2, etc.)"
    ),
    "Property": NodeTypeConfig(
        color="#FF9800",  # Orange
        icon="📊",
        description="Quantitative properties with units (strength, durability)"
    ),
    "Source": NodeTypeConfig(
        color="#9C27B0",  # Purple
        icon="📄",
        description="Source documents and references"
    ),
    
    # New types for material discovery
    "Experiment": NodeTypeConfig(
        color="#EC4899",  # Pink
        icon="🔬",
        description="Laboratory experiments and test records"
    ),
    "ProcessCondition": NodeTypeConfig(
        color="#06B6D4",  # Cyan
        icon="⚙️",
        description="Process conditions (curing, mixing parameters)"
    ),
}


def get_node_color(node_type: str) -> str:
    """
    Get the display color for a node type.
    
    Args:
        node_type: The node type name
        
    Returns:
        Hex color string
    """
    if node_type in NODE_TYPES:
        return NODE_TYPES[node_type].color
    return "#6B7280"  # Default gray


def get_node_icon(node_type: str) -> str:
    """
    Get the icon for a node type.
    
    Args:
        node_type: The node type name
        
    Returns:
        Emoji icon string
    """
    if node_type in NODE_TYPES:
        return NODE_TYPES[node_type].icon
    return "📌"  # Default


def get_node_types() -> list:
    """Get list of all node type names."""
    return list(NODE_TYPES.keys())


def get_node_type_info(node_type: str) -> Optional[NodeTypeConfig]:
    """Get full configuration for a node type."""
    return NODE_TYPES.get(node_type)
