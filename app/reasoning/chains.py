"""Chain of Thought reasoning pipeline."""

from typing import List, Optional, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import get_config
from app.reasoning.models import ReasoningStep
from app.reasoning.prompts import REASONING_SYSTEM_PROMPT, REASONING_USER_PROMPT


class GraphReasoningChain:
    """
    Executes multi-step reasoning over Knowledge Graph data.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the reasoning chain.
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
                "temperature": 0.2,  # Slight creativity for reasoning
                "api_key": self.api_key,
            }
            if self.api_base:
                kwargs["base_url"] = self.api_base
            self._llm = ChatOpenAI(**kwargs)
        return self._llm
    
    def run_reasoning_chain(self, question: str, context: str) -> List[ReasoningStep]:
        """
        Run the Chain-of-Thought reasoning process.
        
        Args:
            question: The user's question or problem.
            context: Relevant context retrieved from the Knowledge Graph.
            
        Returns:
            List of ReasoningStep objects.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", REASONING_SYSTEM_PROMPT),
            ("human", REASONING_USER_PROMPT)
        ])
        
        # Create schema for list of reasoning steps
        steps_schema = {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": ReasoningStep.model_json_schema()
                }
            },
            "required": ["steps"]
        }
        
        structured_llm = self.llm.with_structured_output(steps_schema)
        chain = prompt | structured_llm
        
        try:
            result = chain.invoke({
                "context": context,
                "question": question
            })
            
            steps = []
            for step_data in result.get("steps", []):
                steps.append(ReasoningStep(**step_data))
            return steps
            
        except Exception as e:
            print(f"Reasoning chain failed: {e}")
            return []
