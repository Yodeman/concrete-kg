"""LLM-based extraction logic using LangChain."""

from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import get_config
from app.models.components import MaterialComposition
from app.models.relationships import (
    ComponentRelationship,
    SubstitutionRelationship,
    SynergyRelationship,
)
from app.models.extraction import ExtractionResult
from app.extraction.prompts import (
    get_composition_prompt_with_schema,
    get_substitution_prompt_with_schema,
    get_synergy_prompt_with_schema,
)
from app.pdf_processor import PDFProcessor


class LLMExtractor:
    """
    Extracts structured knowledge from text using LangChain and LLMs.

    Supports extraction of:
    - Material compositions
    - Component relationships
    - Substitution relationships (NEW)
    - Synergy/antagonism relationships (NEW)
    """

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
        """Extract material compositions from text."""
        if not self.is_configured:
            raise ValueError("LLM not configured. Please set OPENAI_API_KEY.")

        prompt_text = get_composition_prompt_with_schema(text)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert in concrete materials science. Extract structured data.",
                ),
                ("human", prompt_text),
            ]
        )

        structured_llm = self.llm.with_structured_output(
            schema=_create_materials_schema(), method="json_schema"
        )

        chain = prompt | structured_llm
        result = chain.invoke({})

        materials = []
        for mat_data in result.get("materials", []):
            materials.append(MaterialComposition(**mat_data))

        return materials

    def extract_substitutions(self, text: str) -> List[SubstitutionRelationship]:
        """Extract substitution relationships from text."""
        if not self.is_configured:
            raise ValueError("LLM not configured. Please set OPENAI_API_KEY.")

        prompt_text = get_substitution_prompt_with_schema(text)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert in concrete materials science. Extract substitution relationships.",
                ),
                ("human", prompt_text),
            ]
        )

        structured_llm = self.llm.with_structured_output(
            schema=_create_substitutions_schema(), method="json_schema"
        )

        chain = prompt | structured_llm
        result = chain.invoke({})

        substitutions = []
        for sub_data in result.get("substitutions", []):
            substitutions.append(SubstitutionRelationship(**sub_data))

        return substitutions

    def extract_synergies(self, text: str) -> List[SynergyRelationship]:
        """Extract synergy/antagonism relationships from text."""
        if not self.is_configured:
            raise ValueError("LLM not configured. Please set OPENAI_API_KEY.")

        prompt_text = get_synergy_prompt_with_schema(text)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert in concrete materials science. Extract synergy and antagonistic relationships.",
                ),
                ("human", prompt_text),
            ]
        )

        structured_llm = self.llm.with_structured_output(
            schema=_create_synergies_schema(), method="json_schema"
        )

        chain = prompt | structured_llm
        result = chain.invoke({})

        synergies = []
        for syn_data in result.get("synergies", []):
            synergies.append(SynergyRelationship(**syn_data))

        return synergies

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
        all_substitutions = []
        all_synergies = []

        for chunk in chunks:
            if len(chunk.strip()) < 100:
                continue

            try:
                materials = self.extract_compositions(chunk)
                all_materials.extend(materials)
            except Exception as e:
                print(f"Error extracting compositions: {e}")

            try:
                substitutions = self.extract_substitutions(chunk)
                all_substitutions.extend(substitutions)
            except Exception as e:
                print(f"Error extracting substitutions: {e}")

            try:
                synergies = self.extract_synergies(chunk)
                all_synergies.extend(synergies)
            except Exception as e:
                print(f"Error extracting synergies: {e}")

        # Deduplicate materials by name
        unique_materials = {}
        for mat in all_materials:
            if mat.material_name not in unique_materials:
                unique_materials[mat.material_name] = mat

        return ExtractionResult(
            materials=list(unique_materials.values()),
            relationships=all_relationships,
            substitutions=all_substitutions,
            synergies=all_synergies,
            source_document=metadata.get("title", "Unknown"),
        )

    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks at paragraph boundaries."""
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
# JSON Schema Helpers
# =============================================================================


def _create_materials_schema() -> dict:
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
                                    "role": {"type": "string"},
                                },
                                "required": ["name", "formula", "role"],
                            },
                        },
                        "properties": {"type": "object"},
                    },
                    "required": ["material_name", "material_type"],
                },
            }
        },
        "required": ["materials"],
    }


def _create_substitutions_schema() -> dict:
    """Create JSON schema for substitutions extraction."""
    return {
        "type": "object",
        "properties": {
            "substitutions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "original": {"type": "string"},
                        "substitute": {"type": "string"},
                        "max_ratio": {"type": "number"},
                        "effects": {"type": "string"},
                        "conditions": {"type": ["string", "null"]},
                    },
                    "required": ["original", "substitute", "max_ratio", "effects"],
                },
            }
        },
        "required": ["substitutions"],
    }


def _create_synergies_schema() -> dict:
    """Create JSON schema for synergies extraction."""
    return {
        "type": "object",
        "properties": {
            "synergies": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "component1": {"type": "string"},
                        "component2": {"type": "string"},
                        "effect_type": {
                            "type": "string",
                            "enum": ["SYNERGY", "ANTAGONISTIC"],
                        },
                        "effect": {"type": "string"},
                        "strength": {"type": "number"},
                        "mechanism": {"type": ["string", "null"]},
                    },
                    "required": [
                        "component1",
                        "component2",
                        "effect_type",
                        "effect",
                        "strength",
                    ],
                },
            }
        },
        "required": ["synergies"],
    }
