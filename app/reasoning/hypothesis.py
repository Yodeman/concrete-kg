"""Hypothesis generation engine."""

from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import get_config
from app.reasoning.models import MaterialHypothesis
from app.reasoning.prompts import HYPOTHESIS_SYSTEM_PROMPT, HYPOTHESIS_USER_PROMPT


class HypothesisGenerator:
    """
    Generates novel material hypotheses based on targets and constraints.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the hypothesis generator.
        """
        config = get_config()
        self.api_key = api_key or config.llm.api_key
        self.model = model or config.llm.model
        self.api_base = config.llm.api_base

        self._llm = None

    @property
    def llm(self) -> ChatOpenAI:
        """Get or create the LLM instance."""
        if self._llm is None:
            kwargs = {
                "model": self.model,
                "temperature": 0.7,  # Higher creativity for hypothesis generation
                "api_key": self.api_key,
            }
            if self.api_base:
                kwargs["base_url"] = self.api_base
            self._llm = ChatOpenAI(**kwargs)
        return self._llm

    def generate_hypothesis(
        self, targets: List[str], constraints: List[str], context: str
    ) -> Optional[MaterialHypothesis]:
        """
        Generate a material hypothesis.

        Args:
            targets: List of target properties (e.g., "high strength").
            constraints: List of constraints (e.g., "no silica fume").
            context: Relevant context from the Knowledge Graph.

        Returns:
            MaterialHypothesis object or None if generation fails.
        """
        prompt = ChatPromptTemplate.from_messages(
            [("system", HYPOTHESIS_SYSTEM_PROMPT), ("human", HYPOTHESIS_USER_PROMPT)]
        )

        structured_llm = self.llm.with_structured_output(MaterialHypothesis)
        chain = prompt | structured_llm

        try:
            # Format inputs
            target_str = ", ".join(targets)
            constraint_str = ", ".join(constraints)

            result = chain.invoke(
                {
                    "context": context,
                    "targets": target_str,
                    "constraints": constraint_str,
                }
            )

            return result

        except Exception as e:
            print(f"Hypothesis generation failed: {e}")
            return None
