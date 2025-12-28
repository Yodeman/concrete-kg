"""Configuration management for the Concrete Knowledge Graph Application."""

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class LLMConfig:
    """Configuration for Language Model settings."""
    
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    api_base: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_BASE"))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    temperature: float = 0.0
    max_tokens: int = 4096
    
    @property
    def is_configured(self) -> bool:
        """Check if LLM is properly configured."""
        return bool(self.api_key and self.api_key != "your-openai-api-key-here")


@dataclass
class Neo4jConfig:
    """Configuration for Neo4j AuraDB connection."""
    
    uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", ""))
    username: str = field(default_factory=lambda: os.getenv("NEO4J_USERNAME", "neo4j"))
    password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))
    
    @property
    def is_configured(self) -> bool:
        """Check if Neo4j is properly configured."""
        return bool(
            self.uri 
            and self.uri != "neo4j+s://your-instance.databases.neo4j.io"
            and self.password 
            and self.password != "your-neo4j-password-here"
        )


@dataclass
class AppConfig:
    """Main application configuration."""
    
    llm: LLMConfig = field(default_factory=LLMConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    demo_mode: bool = field(default_factory=lambda: os.getenv("DEMO_MODE", "true").lower() == "true")
    
    # Graph schema configuration
    node_types: list = field(default_factory=lambda: ["Material", "Component", "Property", "Source"])
    relationship_types: list = field(default_factory=lambda: ["CONTAINS", "AFFECTS", "REACTS_WITH", "CITED_IN"])


# Global configuration instance
config = AppConfig()


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config


def reload_config() -> AppConfig:
    """Reload configuration from environment variables."""
    global config
    load_dotenv(override=True)
    config = AppConfig()
    return config
