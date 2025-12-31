"""Base model with LLM prompt export capabilities."""

from typing import Any, Dict, Type
from pydantic import BaseModel


class LLMExportableModel(BaseModel):
    """
    Base model that can export its schema for LLM prompts.

    Provides utility methods to generate prompt-friendly descriptions
    of the model structure, making it easier to use with LangChain
    structured output.
    """

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {"additionalProperties": False}

    @classmethod
    def to_prompt_schema(cls) -> str:
        """
        Export model as a prompt-friendly schema description.

        Returns:
            Human-readable schema for LLM prompts
        """
        schema = cls.model_json_schema()
        lines = [f"**{cls.__name__}**"]

        if "description" in schema:
            lines.append(f"{schema['description']}")

        lines.append("\nFields:")

        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        for field_name, field_info in properties.items():
            req_marker = "*" if field_name in required else ""
            field_type = field_info.get("type", "any")
            description = field_info.get("description", "")

            lines.append(f"  - {field_name}{req_marker} ({field_type}): {description}")

        return "\n".join(lines)

    @classmethod
    def get_field_descriptions(cls) -> Dict[str, str]:
        """
        Get a dictionary of field names to descriptions.

        Returns:
            Dict mapping field names to their descriptions
        """
        schema = cls.model_json_schema()
        properties = schema.get("properties", {})

        return {
            field_name: field_info.get("description", "")
            for field_name, field_info in properties.items()
        }

    @classmethod
    def to_extraction_prompt(cls) -> str:
        """
        Generate an extraction prompt section for this model.

        Returns:
            Prompt text describing what to extract
        """
        schema = cls.model_json_schema()
        fields = []

        for field_name, field_info in schema.get("properties", {}).items():
            desc = field_info.get("description", field_name)
            fields.append(f"- {field_name}: {desc}")

        return f"Extract {cls.__name__} with:\n" + "\n".join(fields)
