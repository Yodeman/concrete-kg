"""Reasoning models for hypothesis generation and validation."""

from typing import List, Optional, Literal
from pydantic import Field

from app.models.base import LLMExportableModel
from app.models.components import MaterialComposition
from app.models.properties import QuantitativeProperty


class ReasoningStep(LLMExportableModel):
    """
    A single step in a Chain-of-Thought reasoning process.
    """

    step_number: int = Field(
        ..., description="The sequence number of this reasoning step"
    )
    thought: str = Field(
        ..., description="The reasoning thought or deduction made in this step"
    )
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="IDs of Knowledge Graph nodes or edges that support this thought",
    )
    conclusion: str = Field(
        ..., description="The intermediate conclusion reached in this step"
    )


class ValidationStatus(LLMExportableModel):
    """
    Status of a hypothesis validation check.
    """

    is_valid: bool = Field(
        ..., description="Whether the hypothesis is considered valid"
    )
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    issues: List[str] = Field(
        default_factory=list,
        description="List of potential issues or contradictions found",
    )
    supporting_evidence: List[str] = Field(
        default_factory=list,
        description="List of supporting facts from the knowledge graph",
    )


class MaterialHypothesis(LLMExportableModel):
    """
    A generated hypothesis for a new material composition.
    """

    title: str = Field(..., description="A descriptive title for the hypothesis")
    description: str = Field(
        ...,
        description="Detailed description of the proposed material and its intended benefits",
    )
    target_properties: List[str] = Field(
        default_factory=list,
        description="List of target properties this hypothesis aims to optimize (e.g., 'high sulfate resistance')",
    )
    proposed_composition: MaterialComposition = Field(
        ..., description="The proposed chemical/material composition"
    )
    predicted_properties: List[QuantitativeProperty] = Field(
        default_factory=list,
        description="Predicted quantitative properties of the new material",
    )
    reasoning_chain: List[ReasoningStep] = Field(
        default_factory=list,
        description="The step-by-step reasoning that led to this hypothesis",
    )
    confidence_score: float = Field(
        ..., ge=0, le=1, description="Overall confidence in the hypothesis (0-1)"
    )
    contradictions: List[str] = Field(
        default_factory=list,
        description="Known contradictions or risks associated with this hypothesis",
    )
