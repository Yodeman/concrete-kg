"""LLM-based extraction pipeline for concrete material compositions."""

from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .config import get_config
from .pdf_processor import PDFProcessor


# =============================================================================
# Pydantic Models for Structured Extraction
# =============================================================================

class ChemicalComponent(BaseModel):
    """Represents a chemical component in a concrete material."""
    
    name: str = Field(description="Common name of the component (e.g., 'Calcium Oxide', 'Silica')")
    formula: str = Field(description="Chemical formula (e.g., 'CaO', 'SiO2')")
    percentage: Optional[float] = Field(default=None, description="Percentage by weight in the material")
    role: str = Field(description="Role in concrete (e.g., 'binder', 'aggregate', 'admixture', 'hydration product')")


class MaterialComposition(BaseModel):
    """Represents a concrete material and its composition."""
    
    material_name: str = Field(description="Name of the material (e.g., 'Portland Cement', 'Fly Ash')")
    material_type: str = Field(description="Type classification (e.g., 'cement', 'supplementary cite material', 'aggregate')")
    components: List[ChemicalComponent] = Field(default_factory=list, description="List of chemical components")
    properties: dict = Field(default_factory=dict, description="Physical/mechanical properties mentioned")


class ComponentRelationship(BaseModel):
    """Represents a relationship between components."""
    
    source: str = Field(description="Source component name or formula")
    target: str = Field(description="Target component name or formula")
    relationship_type: str = Field(description="Type of relationship: CONTAINS, AFFECTS, REACTS_WITH, PRODUCES")
    description: str = Field(description="Description of the relationship")
    strength: Optional[float] = Field(default=None, description="Relationship strength/importance (0-1)")


class ExtractionResult(BaseModel):
    """Complete extraction result from a document."""
    
    materials: List[MaterialComposition] = Field(default_factory=list)
    relationships: List[ComponentRelationship] = Field(default_factory=list)
    source_document: Optional[str] = Field(default=None, description="Source document identifier")


# =============================================================================
# Extraction Prompts
# =============================================================================

COMPOSITION_EXTRACTION_PROMPT = """You are an expert in concrete materials science and civil engineering.
Analyze the following text from a research paper and extract information about concrete material compositions.

Focus on:
1. Identifying concrete materials (cements, SCMs, aggregates, admixtures)
2. Their chemical compositions (oxides, compounds)
3. Percentages and proportions when mentioned
4. Roles of each component in concrete

Text to analyze:
{text}

Extract all materials and their chemical compositions found in the text."""


RELATIONSHIP_EXTRACTION_PROMPT = """You are an expert in concrete materials science.
Analyze the following text and identify relationships between chemical components in concrete.

Types of relationships to look for:
- CONTAINS: A material contains a component
- AFFECTS: A component affects a property (e.g., strength, durability)
- REACTS_WITH: Components that react with each other
- PRODUCES: Reactions that produce new compounds

Text to analyze:
{text}

Extract all relationships between components found in the text."""


# =============================================================================
# LLM Extractor Class
# =============================================================================

class LLMExtractor:
    """Extracts structured knowledge from text using LangChain and LLMs."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the LLM extractor.
        
        Args:
            api_key: OpenAI API key (uses config if not provided)
            model: Model name (uses config if not provided)
        """
        config = get_config()
        self.api_key = api_key or config.llm.api_key
        self.model = model or config.llm.model
        self.api_base = config.llm.api_base
        
        self._llm = None
        self._pdf_processor = PDFProcessor()
    
    @property
    def llm(self) -> ChatOpenAI:
        """Get or create the LLM instance."""
        if self._llm is None:
            kwargs = {
                "model": self.model,
                "temperature": 0,
                "api_key": self.api_key,
            }
            if self.api_base:
                kwargs["base_url"] = self.api_base
            self._llm = ChatOpenAI(**kwargs)
        return self._llm
    
    @property
    def is_configured(self) -> bool:
        """Check if the extractor is properly configured."""
        return bool(self.api_key and self.api_key != "your-openai-api-key-here")
    
    def extract_compositions(self, text: str) -> List[MaterialComposition]:
        """
        Extract material compositions from text.
        
        Args:
            text: Text content to analyze
            
        Returns:
            List of MaterialComposition objects
        """
        if not self.is_configured:
            raise ValueError("LLM not configured. Please set OPENAI_API_KEY.")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in concrete materials science. Extract structured data."),
            ("human", COMPOSITION_EXTRACTION_PROMPT)
        ])
        
        # Use structured output
        structured_llm = self.llm.with_structured_output(
            schema=create_materials_schema(),
            method="json_schema"
        )
        
        chain = prompt | structured_llm
        result = chain.invoke({"text": text})
        
        return result.get("materials", []) if isinstance(result, dict) else []
    
    def extract_relationships(self, text: str) -> List[ComponentRelationship]:
        """
        Extract relationships between components from text.
        
        Args:
            text: Text content to analyze
            
        Returns:
            List of ComponentRelationship objects
        """
        if not self.is_configured:
            raise ValueError("LLM not configured. Please set OPENAI_API_KEY.")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in concrete materials science. Extract relationships."),
            ("human", RELATIONSHIP_EXTRACTION_PROMPT)
        ])
        
        structured_llm = self.llm.with_structured_output(
            schema=create_relationships_schema(),
            method="json_schema"
        )
        
        chain = prompt | structured_llm
        result = chain.invoke({"text": text})
        
        return result.get("relationships", []) if isinstance(result, dict) else []
    
    def process_document(self, pdf_source, chunk_size: int = 4000) -> ExtractionResult:
        """
        Process a complete PDF document and extract all knowledge.
        
        Args:
            pdf_source: PDF file path, bytes, or file-like object
            chunk_size: Maximum characters per chunk for processing
            
        Returns:
            ExtractionResult with all extracted data
        """
        if not self.is_configured:
            raise ValueError("LLM not configured. Please set OPENAI_API_KEY.")
        
        # Extract text from PDF
        text = self._pdf_processor.extract_text(pdf_source)
        metadata = self._pdf_processor.get_metadata(pdf_source)
        
        # Process in chunks if text is too long
        chunks = self._chunk_text(text, chunk_size)
        
        all_materials = []
        all_relationships = []
        
        for chunk in chunks:
            if len(chunk.strip()) < 100:  # Skip very short chunks
                continue
            
            try:
                materials = self.extract_compositions(chunk)
                all_materials.extend(materials)
            except Exception as e:
                print(f"Error extracting compositions: {e}")
            
            try:
                relationships = self.extract_relationships(chunk)
                all_relationships.extend(relationships)
            except Exception as e:
                print(f"Error extracting relationships: {e}")
        
        # Deduplicate materials by name
        unique_materials = {}
        for mat in all_materials:
            if isinstance(mat, dict):
                mat = MaterialComposition(**mat)
            if mat.material_name not in unique_materials:
                unique_materials[mat.material_name] = mat
            else:
                # Merge components
                existing = unique_materials[mat.material_name]
                existing_formulas = {c.formula for c in existing.components}
                for comp in mat.components:
                    if comp.formula not in existing_formulas:
                        existing.components.append(comp)
        
        return ExtractionResult(
            materials=list(unique_materials.values()),
            relationships=all_relationships,
            source_document=metadata.get("title", "Unknown")
        )
    
    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks, trying to break at paragraph boundaries."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        paragraphs = text.split("\n\n")
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks


# =============================================================================
# Schema Helpers for Structured Output
# =============================================================================

def create_materials_schema() -> dict:
    """Create JSON schema for materials extraction."""
    return {
        "type": "object",
        "properties": {
            "materials": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "material_name": {"type": "string"},
                        "material_type": {"type": "string"},
                        "components": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "formula": {"type": "string"},
                                    "percentage": {"type": ["number", "null"]},
                                    "role": {"type": "string"}
                                },
                                "required": ["name", "formula", "role"]
                            }
                        },
                        "properties": {"type": "object"}
                    },
                    "required": ["material_name", "material_type", "components"]
                }
            }
        },
        "required": ["materials"]
    }


def create_relationships_schema() -> dict:
    """Create JSON schema for relationships extraction."""
    return {
        "type": "object",
        "properties": {
            "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "target": {"type": "string"},
                        "relationship_type": {"type": "string"},
                        "description": {"type": "string"},
                        "strength": {"type": ["number", "null"]}
                    },
                    "required": ["source", "target", "relationship_type", "description"]
                }
            }
        },
        "required": ["relationships"]
    }


# =============================================================================
# Demo Data for Testing
# =============================================================================

def get_demo_extraction_result() -> ExtractionResult:
    """Get sample extraction result for demo mode."""
    portland_cement = MaterialComposition(
        material_name="Portland Cement",
        material_type="cement",
        components=[
            ChemicalComponent(name="Calcium Oxide", formula="CaO", percentage=65.0, role="binder"),
            ChemicalComponent(name="Silicon Dioxide", formula="SiO2", percentage=22.0, role="binder"),
            ChemicalComponent(name="Aluminum Oxide", formula="Al2O3", percentage=5.5, role="modifier"),
            ChemicalComponent(name="Iron Oxide", formula="Fe2O3", percentage=3.5, role="flux"),
            ChemicalComponent(name="Magnesium Oxide", formula="MgO", percentage=2.0, role="minor"),
            ChemicalComponent(name="Sulfur Trioxide", formula="SO3", percentage=2.0, role="setting regulator"),
        ],
        properties={"compressive_strength": "40-60 MPa", "setting_time": "45-375 min"}
    )
    
    fly_ash = MaterialComposition(
        material_name="Fly Ash (Class F)",
        material_type="supplementary cite material",
        components=[
            ChemicalComponent(name="Silicon Dioxide", formula="SiO2", percentage=55.0, role="pozzolanic"),
            ChemicalComponent(name="Aluminum Oxide", formula="Al2O3", percentage=25.0, role="pozzolanic"),
            ChemicalComponent(name="Iron Oxide", formula="Fe2O3", percentage=10.0, role="inert"),
            ChemicalComponent(name="Calcium Oxide", formula="CaO", percentage=5.0, role="reactive"),
        ],
        properties={"fineness": "300-400 m²/kg", "loss_on_ignition": "<6%"}
    )
    
    silica_fume = MaterialComposition(
        material_name="Silica Fume",
        material_type="supplementary cementitious material",
        components=[
            ChemicalComponent(name="Silicon Dioxide", formula="SiO2", percentage=92.0, role="pozzolanic"),
            ChemicalComponent(name="Carbon", formula="C", percentage=3.0, role="impurity"),
            ChemicalComponent(name="Iron Oxide", formula="Fe2O3", percentage=1.5, role="minor"),
        ],
        properties={"specific_surface": "15000-25000 m²/kg", "particle_size": "0.1-0.3 µm"}
    )
    
    relationships = [
        ComponentRelationship(
            source="CaO", target="SiO2", relationship_type="REACTS_WITH",
            description="Calcium oxide reacts with silica during hydration to form C-S-H gel",
            strength=0.95
        ),
        ComponentRelationship(
            source="C-S-H", target="Compressive Strength", relationship_type="AFFECTS",
            description="Calcium silicate hydrate is the primary contributor to concrete strength",
            strength=0.9
        ),
        ComponentRelationship(
            source="Al2O3", target="Setting Time", relationship_type="AFFECTS",
            description="Aluminum oxide content influences the setting behavior of cement",
            strength=0.7
        ),
        ComponentRelationship(
            source="SO3", target="Setting Time", relationship_type="AFFECTS",
            description="Sulfur trioxide (as gypsum) controls setting time",
            strength=0.85
        ),
        ComponentRelationship(
            source="SiO2", target="Durability", relationship_type="AFFECTS",
            description="Silica from pozzolans improves long-term durability",
            strength=0.8
        ),
        ComponentRelationship(
            source="Portland Cement", target="CaO", relationship_type="CONTAINS",
            description="Portland cement primarily consists of calcium oxide",
            strength=1.0
        ),
        ComponentRelationship(
            source="Fe2O3", target="Color", relationship_type="AFFECTS",
            description="Iron oxide contributes to the gray color of cement",
            strength=0.6
        ),
    ]
    
    return ExtractionResult(
        materials=[portland_cement, fly_ash, silica_fume],
        relationships=relationships,
        source_document="Demo: Concrete Materials Composition"
    )
