"""Component and Material models for concrete compositions."""

from typing import Dict, List, Optional, Any
from pydantic import Field

from app.models.base import LLMExportableModel


class ChemicalComponent(LLMExportableModel):
    """
    Represents a chemical component in a concrete material.

    Chemical components are the building blocks of concrete materials,
    including oxides (CaO, SiO2), compounds, and additives.
    """

    name: str = Field(
        ...,
        description="Common name of the component (e.g., 'Calcium Oxide', 'Silica')",
    )
    formula: str = Field(
        ..., description="Chemical formula (e.g., 'CaO', 'SiO2', 'Al2O3')"
    )
    percentage: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Percentage by weight in the material (0-100)",
    )
    role: str = Field(
        ...,
        description="Role in concrete: 'binder', 'aggregate', 'admixture', 'hydration_product', 'filler'",
    )
    category: Optional[str] = Field(
        default=None, description="Category: 'oxide', 'mineral', 'organic', 'inorganic'"
    )


class MaterialComposition(LLMExportableModel):
    """
    Represents a concrete material and its chemical composition.

    Materials are complete mixtures or substances used in concrete,
    such as Portland Cement, Fly Ash, or Silica Fume.
    """

    material_name: str = Field(
        ...,
        description="Name of the material (e.g., 'Portland Cement', 'Fly Ash Class F')",
    )
    material_type: str = Field(
        ...,
        description="Type: 'cement', 'supplementary_cite_material', 'aggregate', 'admixture', 'water'",
    )
    components: List[ChemicalComponent] = Field(
        default_factory=list, description="List of chemical components in this material"
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Physical/mechanical properties (e.g., specific_gravity, fineness)",
    )
    source: Optional[str] = Field(
        default=None, description="Source document or standard (e.g., 'ASTM C150')"
    )
