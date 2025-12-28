"""Property and condition models for quantitative data."""

from typing import Optional, Literal
from pydantic import Field

from app.models.base import LLMExportableModel


class QuantitativeProperty(LLMExportableModel):
    """
    Represents a quantitative property with units and uncertainty.
    
    Used for concrete properties like compressive strength, durability,
    setting time, etc. with full measurement metadata.
    """
    
    name: str = Field(
        ...,
        description="Property name (e.g., 'Compressive Strength', 'Setting Time')"
    )
    value: float = Field(
        ...,
        description="Numeric value of the property"
    )
    unit: str = Field(
        ...,
        description="Unit of measurement (e.g., 'MPa', 'minutes', 'kg/m³')"
    )
    conditions: Optional[str] = Field(
        default=None,
        description="Measurement conditions (e.g., 'at 28 days', 'at 20°C')"
    )
    min_value: Optional[float] = Field(
        default=None,
        description="Minimum acceptable value (for specifications)"
    )
    max_value: Optional[float] = Field(
        default=None,
        description="Maximum acceptable value (for specifications)"
    )
    uncertainty: Optional[float] = Field(
        default=None,
        description="Measurement uncertainty (± value in same units)"
    )
    test_method: Optional[str] = Field(
        default=None,
        description="Test method standard (e.g., 'ASTM C39')"
    )


class ProcessCondition(LLMExportableModel):
    """
    Represents a process or environmental condition.
    
    Used for curing conditions, mixing parameters, and other
    process variables that affect material properties.
    """
    
    condition_type: Literal["curing", "mixing", "environmental", "testing"] = Field(
        ...,
        description="Type of condition"
    )
    parameter: str = Field(
        ...,
        description="Parameter name (e.g., 'temperature', 'humidity', 'duration')"
    )
    value: float = Field(
        ...,
        description="Numeric value"
    )
    unit: str = Field(
        ...,
        description="Unit of measurement (e.g., '°C', '%', 'hours')"
    )
    min_value: Optional[float] = Field(
        default=None,
        description="Minimum value for valid range"
    )
    max_value: Optional[float] = Field(
        default=None,
        description="Maximum value for valid range"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Additional notes or requirements"
    )


class ExperimentRecord(LLMExportableModel):
    """
    Represents a laboratory experiment or test record.
    
    Links materials to their measured properties under specific
    conditions, enabling validation and model training.
    """
    
    experiment_id: str = Field(
        ...,
        description="Unique identifier for the experiment"
    )
    material_name: str = Field(
        ...,
        description="Material being tested"
    )
    date: Optional[str] = Field(
        default=None,
        description="Date of experiment (ISO format)"
    )
    conditions: list = Field(
        default_factory=list,
        description="List of ProcessCondition objects"
    )
    results: list = Field(
        default_factory=list,
        description="List of QuantitativeProperty results"
    )
    source: Optional[str] = Field(
        default=None,
        description="Source document or lab reference"
    )
    validated: bool = Field(
        default=False,
        description="Whether results have been validated"
    )
