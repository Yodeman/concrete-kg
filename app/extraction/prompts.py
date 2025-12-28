"""Extraction prompts for LLM-based knowledge extraction.

These prompts are designed to work with the Pydantic models in app/models/
and can be combined with model schemas for structured extraction.
"""

from app.models.components import ChemicalComponent, MaterialComposition
from app.models.relationships import SubstitutionRelationship, SynergyRelationship


# =============================================================================
# Core Composition Extraction
# =============================================================================

COMPOSITION_EXTRACTION_PROMPT = """You are an expert in concrete materials science and civil engineering.
Analyze the following text from a research paper and extract information about concrete material compositions.

Focus on:
1. Identifying concrete materials (cements, SCMs, aggregates, admixtures)
2. Their chemical compositions (oxides, compounds)
3. Percentages and proportions when mentioned
4. Roles of each component in concrete

{schema}

Text to analyze:
{text}

Extract all materials and their chemical compositions found in the text."""


# =============================================================================
# Substitution Extraction (NEW)
# =============================================================================

SUBSTITUTION_EXTRACTION_PROMPT = """You are an expert in concrete materials science.
Analyze the following text and identify material SUBSTITUTIONS mentioned.

Look for phrases indicating one material can replace another:
- "can replace", "substitute for", "alternative to"
- "replacement of X with Y", "X can be substituted by Y"
- "partial replacement", "up to X% replacement"

For each substitution, extract:
- Original material being replaced
- Substitute material
- Maximum replacement ratio (as decimal, e.g., 0.3 for 30%)
- Effects on properties (strength, durability, workability)
- Conditions when substitution is valid

{schema}

Text to analyze:
{text}

Extract all substitution relationships found in the text."""


# =============================================================================
# Synergy/Antagonism Extraction (NEW)
# =============================================================================

SYNERGY_EXTRACTION_PROMPT = """You are an expert in concrete materials science.
Analyze the following text and identify SYNERGISTIC or ANTAGONISTIC effects between components.

SYNERGISTIC effects (positive combined effects):
- "combined with X improves", "enhanced by", "works well with"
- "synergistic effect", "complementary action"

ANTAGONISTIC effects (negative combined effects):
- "should not be combined with", "inhibits", "reduces when mixed"
- "incompatible with", "adversely affects", "causes degradation"

For each interaction, extract:
- Two components involved
- Whether it's SYNERGY or ANTAGONISTIC
- Description of the effect
- Strength of the effect (0-1 scale)
- Mechanism if mentioned

{schema}

Text to analyze:
{text}

Extract all synergistic and antagonistic relationships found in the text."""


# =============================================================================
# Quantitative Property Extraction (NEW)
# =============================================================================

QUANTITATIVE_PROPERTY_PROMPT = """You are an expert in concrete materials science.
Analyze the following text and extract QUANTITATIVE PROPERTIES of materials.

Focus on measurable properties with numeric values:
- Mechanical: compressive strength, tensile strength, flexural strength
- Physical: density, porosity, water absorption
- Durability: chloride penetration, carbonation depth
- Fresh properties: slump, setting time, workability

For each property, extract:
- Property name
- Numeric value
- Unit of measurement
- Test conditions (age, temperature, etc.)
- Uncertainty if mentioned

{schema}

Text to analyze:
{text}

Extract all quantitative properties found in the text."""


# =============================================================================
# Schema Generators
# =============================================================================

def get_composition_prompt_with_schema(text: str) -> str:
    """Generate composition extraction prompt with model schema."""
    schema = MaterialComposition.to_prompt_schema()
    return COMPOSITION_EXTRACTION_PROMPT.format(schema=schema, text=text)


def get_substitution_prompt_with_schema(text: str) -> str:
    """Generate substitution extraction prompt with model schema."""
    schema = SubstitutionRelationship.to_prompt_schema()
    return SUBSTITUTION_EXTRACTION_PROMPT.format(schema=schema, text=text)


def get_synergy_prompt_with_schema(text: str) -> str:
    """Generate synergy extraction prompt with model schema."""
    schema = SynergyRelationship.to_prompt_schema()
    return SYNERGY_EXTRACTION_PROMPT.format(schema=schema, text=text)
