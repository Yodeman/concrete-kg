"""Prompts for reasoning and hypothesis generation."""

from app.reasoning.models import MaterialHypothesis, ReasoningStep

# =============================================================================
# Chain of Thought Reasoning
# =============================================================================

REASONING_SYSTEM_PROMPT = """You are an expert material scientist AI assistant.
Your goal is to reason through material science problems step-by-step using a Knowledge Graph.

You should:
1. Break down complex problems into logical steps.
2. Cite evidence from the provided context (Knowledge Graph data).
3. Identify substitutions, synergies, and potential antagonisms.
4. Formulate a clear conclusion.

Context from Knowledge Graph:
{context}
"""

REASONING_USER_PROMPT = """Question: {question}

Reason through this step-by-step and provide a structured Chain-of-Thought response.
"""

# =============================================================================
# Hypothesis Generation
# =============================================================================

HYPOTHESIS_SYSTEM_PROMPT = """You are an innovative concrete material scientist.
Your goal is to propose NOVEL material compositions (hypotheses) to meet specific target properties.

Guidelines:
1. Analyze the target properties and constraints.
2. Use the provided Knowledge Graph context to find:
   - Substitutes that improve target properties.
   - Synergistic combinations.
   - Materials to avoid (antagonistic).
3. Propose a specific composition (percentages).
4. Predict the properties of this new mix.
5. Check for contradictions.

Context from Knowledge Graph:
{context}

Target Properties: {targets}
Constraints: {constraints}
"""

HYPOTHESIS_USER_PROMPT = """Generate a material hypothesis based on the targets and constraints.
"""


def get_reasoning_prompt_with_schema(question: str, context: str) -> str:
    """
    Format the reasoning prompt with context and schema.
    """
    schema = ReasoningStep.to_prompt_schema()
    return f"""{REASONING_USER_PROMPT.format(question=question)}

Output Format:
Produce a list of ReasoningSteps.
{schema}
"""


def get_hypothesis_prompt_with_schema(
    targets: str, constraints: str, context: str
) -> str:
    """
    Format the hypothesis generation prompt.
    """
    # schema = MaterialHypothesis.to_prompt_schema()
    # Note: We usually use the structured output API, but this is for text-based prompts if needed.
    return HYPOTHESIS_USER_PROMPT
