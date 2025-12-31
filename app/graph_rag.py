"""Graph RAG (Retrieval-Augmented Generation) engine for knowledge graph querying."""

from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import get_config
from .knowledge_graph import KnowledgeGraph


class GraphRAG:
    """
    Graph RAG engine for natural language querying over the knowledge graph.

    Combines:
    - Cypher query generation for structured retrieval
    - Vector similarity for semantic search (when Neo4j is configured)
    - LLM-based answer synthesis
    """

    def __init__(self, knowledge_graph: KnowledgeGraph):
        """
        Initialize Graph RAG engine.

        Args:
            knowledge_graph: KnowledgeGraph instance to query
        """
        self.kg = knowledge_graph
        config = get_config()

        self._llm = None
        self._cypher_chain = None
        self._vector_store = None

        self.api_key = config.llm.api_key
        self.model = config.llm.model
        self.api_base = config.llm.api_base

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
        """Check if Graph RAG is properly configured."""
        return bool(self.api_key and self.api_key != "your-openai-api-key-here")

    def _init_cypher_chain(self):
        """Initialize the Cypher QA chain for Neo4j queries."""
        if not self.kg.use_neo4j or not self.kg.neo4j_graph:
            return None

        try:
            from langchain_neo4j import GraphCypherQAChain

            self._cypher_chain = GraphCypherQAChain.from_llm(
                llm=self.llm,
                graph=self.kg.neo4j_graph,
                verbose=True,
                return_intermediate_steps=True,
            )
            return self._cypher_chain
        except ImportError:
            print("langchain-neo4j not installed for Cypher chain")
            return None
        except Exception as e:
            print(f"Error initializing Cypher chain: {e}")
            return None

    def _init_vector_store(self):
        """Initialize vector store for semantic search."""
        if not self.kg.use_neo4j or not self.kg.neo4j_graph:
            return None

        try:
            from langchain_neo4j import Neo4jVector
            from langchain_openai import OpenAIEmbeddings

            config = get_config()
            embeddings = OpenAIEmbeddings(api_key=self.api_key)

            # Create vector index on node labels/descriptions
            self._vector_store = Neo4jVector.from_existing_graph(
                embedding=embeddings,
                url=config.neo4j.uri,
                username=config.neo4j.username,
                password=config.neo4j.password,
                index_name="node_embeddings",
                node_label="Component",
                text_node_properties=["label", "formula", "role"],
                embedding_node_property="embedding",
            )
            return self._vector_store
        except ImportError:
            print("Required packages not installed for vector store")
            return None
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            return None

    # =========================================================================
    # Query Methods
    # =========================================================================

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Answer a natural language question about the knowledge graph.

        Args:
            question: Natural language question

        Returns:
            Dictionary with answer and supporting context
        """
        if not self.is_configured:
            return self._ask_without_llm(question)

        # Try Cypher-based retrieval first if Neo4j is available
        if self.kg.use_neo4j and self.kg.neo4j_graph:
            try:
                return self._ask_with_cypher(question)
            except Exception as e:
                print(f"Cypher query failed, falling back to in-memory: {e}")

        # Fall back to in-memory graph search + LLM
        return self._ask_with_memory_graph(question)

    def _ask_without_llm(self, question: str) -> Dict[str, Any]:
        """Answer question using only graph structure (no LLM)."""
        question_lower = question.lower()

        # Simple keyword-based search
        results = []

        # Check for material queries
        if "component" in question_lower or "contain" in question_lower:
            materials = self.kg.get_nodes_by_type("Material")
            for mat in materials:
                neighbors = self.kg.get_neighbors(mat["id"])
                components = [n for n in neighbors if n.get("node_type") == "Component"]
                if components:
                    results.append(
                        {
                            "material": mat["label"],
                            "components": [c["label"] for c in components],
                        }
                    )

        # Check for specific component queries
        components = self.kg.get_nodes_by_type("Component")
        for comp in components:
            if (
                comp.get("formula", "").lower() in question_lower
                or comp.get("label", "").lower() in question_lower
            ):
                neighbors = self.kg.get_neighbors(comp["id"])
                results.append(
                    {
                        "component": comp["label"],
                        "formula": comp.get("formula"),
                        "related": [
                            n.get("label", n.get("id", "Unknown")) for n in neighbors
                        ],
                    }
                )

        if results:
            return {
                "answer": f"Found {len(results)} relevant items in the knowledge graph.",
                "context": results,
                "method": "keyword_search",
            }

        return {
            "answer": "Unable to answer without LLM. Please configure your API key.",
            "context": [],
            "method": "none",
        }

    def _ask_with_cypher(self, question: str) -> Dict[str, Any]:
        """Answer question using Neo4j Cypher queries."""
        if self._cypher_chain is None:
            self._init_cypher_chain()

        if self._cypher_chain is None:
            raise RuntimeError("Cypher chain not initialized")

        result = self._cypher_chain.invoke({"query": question})

        return {
            "answer": result.get("result", "No answer found"),
            "context": result.get("intermediate_steps", []),
            "method": "cypher_qa",
        }

    def _ask_with_memory_graph(self, question: str) -> Dict[str, Any]:
        """Answer question using in-memory graph and LLM."""
        # Build context from graph
        context = self._build_context_for_question(question)

        # Create prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert in concrete materials science. 
Answer questions based on the provided knowledge graph context.
If the context doesn't contain the answer, say so clearly.
Be concise and cite specific components or materials when relevant.""",
                ),
                (
                    "human",
                    """Knowledge Graph Context:
{context}

Question: {question}

Answer:""",
                ),
            ]
        )

        chain = prompt | self.llm
        response = chain.invoke({"context": context, "question": question})

        return {
            "answer": response.content,
            "context": context,
            "method": "memory_graph_llm",
        }

    def _build_context_for_question(self, question: str) -> str:
        """Build relevant context from the graph for a question."""
        context_parts = []
        question_lower = question.lower()

        # Get relevant materials
        materials = self.kg.get_nodes_by_type("Material")
        for mat in materials:
            if mat["label"].lower() in question_lower or any(
                word in question_lower for word in mat["label"].lower().split()
            ):
                neighbors = self.kg.get_neighbors(mat["id"])
                components = [
                    n
                    for n in neighbors
                    if "percentage" in n or n.get("node_type") == "Component"
                ]
                if components:
                    comp_strs = []
                    for c in components:
                        pct = c.get("percentage", "")
                        formula = c.get("formula", "")
                        comp_strs.append(
                            f"{c['label']} ({formula})" + (f": {pct}%" if pct else "")
                        )
                    context_parts.append(
                        f"Material: {mat['label']}\nComponents: {', '.join(comp_strs)}"
                    )

        # Get relevant components
        components = self.kg.get_nodes_by_type("Component")
        for comp in components:
            if (
                comp.get("formula", "").lower() in question_lower
                or comp.get("label", "").lower() in question_lower
            ):
                context_parts.append(
                    f"Component: {comp['label']} ({comp.get('formula', '')}), Role: {comp.get('role', 'unknown')}"
                )

        # Get relevant relationships
        for source, target, data in self.kg.graph.edges(data=True):
            if data.get("relationship_type") in ["AFFECTS", "REACTS_WITH"]:
                desc = data.get("description", "")
                if any(word in question_lower for word in desc.lower().split()[:5]):
                    context_parts.append(
                        f"Relationship: {source} {data['relationship_type']} {target}: {desc}"
                    )

        # If no specific matches, include general context
        if not context_parts:
            # Include all materials with components
            for mat in materials[:5]:  # Limit to avoid context overflow
                neighbors = self.kg.get_neighbors(mat["id"])
                components = [n for n in neighbors if n.get("node_type") == "Component"]
                if components:
                    comp_names = [
                        f"{c['label']} ({c.get('formula', '')})" for c in components[:5]
                    ]
                    context_parts.append(
                        f"Material: {mat['label']} contains: {', '.join(comp_names)}"
                    )

        return (
            "\n\n".join(context_parts)
            if context_parts
            else "No relevant context found in knowledge graph."
        )

    # =========================================================================
    # Specialized Queries
    # =========================================================================

    def get_related_components(self, component: str) -> List[Dict[str, Any]]:
        """
        Find components related to the given component.

        Args:
            component: Component name or formula

        Returns:
            List of related components with relationship info
        """
        # Find the component node
        node_id = None
        for nid, data in self.kg.graph.nodes(data=True):
            if data.get("formula") == component or data.get("label") == component:
                node_id = nid
                break

        if not node_id:
            return []

        # Get 2-hop neighborhood
        subgraph = self.kg.get_subgraph(node_id, depth=2)

        related = []
        for nid, data in subgraph.nodes(data=True):
            if nid != node_id:
                # Find path to this node
                try:
                    import networkx as nx

                    paths = list(
                        nx.all_simple_paths(self.kg.graph, node_id, nid, cutoff=2)
                    )
                    if paths:
                        path = paths[0]
                        # Get relationship
                        if len(path) >= 2:
                            edge_data = self.kg.graph.edges[path[0], path[1]]
                            related.append(
                                {
                                    "id": nid,
                                    "label": data.get("label", nid),
                                    "type": data.get("node_type", "Unknown"),
                                    "relationship": edge_data.get(
                                        "relationship_type", "RELATED"
                                    ),
                                    "description": edge_data.get("description", ""),
                                }
                            )
                except Exception:
                    pass

        return related

    def get_composition_context(self, material: str) -> str:
        """
        Get detailed composition context for a material.

        Args:
            material: Material name

        Returns:
            Formatted composition context string
        """
        # Find material node
        material_node = None
        for nid, data in self.kg.graph.nodes(data=True):
            if (
                data.get("node_type") == "Material"
                and material.lower() in data.get("label", "").lower()
            ):
                material_node = nid
                break

        if not material_node:
            return f"Material '{material}' not found in knowledge graph."

        data = self.kg.graph.nodes[material_node]
        lines = [f"# {data.get('label', material)}", ""]

        # Get properties
        props = {
            k: v
            for k, v in data.items()
            if k not in ["label", "node_type", "color", "id"]
        }
        if props:
            lines.append("## Properties")
            for k, v in props.items():
                lines.append(f"- {k}: {v}")
            lines.append("")

        # Get components
        neighbors = self.kg.get_neighbors(material_node)
        components = [n for n in neighbors if n.get("node_type") == "Component"]

        if components:
            lines.append("## Chemical Composition")
            lines.append("| Component | Formula | Percentage | Role |")
            lines.append("|-----------|---------|------------|------|")
            for comp in sorted(
                components, key=lambda x: x.get("percentage", 0) or 0, reverse=True
            ):
                pct = comp.get("percentage", "-")
                lines.append(
                    f"| {comp['label']} | {comp.get('formula', '-')} | {pct}% | {comp.get('role', '-')} |"
                )

        return "\n".join(lines)

    def search_by_property(self, property_query: str) -> List[Dict[str, Any]]:
        """
        Search for components that affect a property.

        Args:
            property_query: Property name or description

        Returns:
            List of components affecting the property
        """
        results = []
        query_lower = property_query.lower()

        for source, target, data in self.kg.graph.edges(data=True):
            if data.get("relationship_type") == "AFFECTS":
                target_data = self.kg.graph.nodes.get(target, {})
                desc = data.get("description", "").lower()

                if (
                    query_lower in target_data.get("label", "").lower()
                    or query_lower in desc
                ):
                    source_data = self.kg.graph.nodes.get(source, {})
                    results.append(
                        {
                            "component": source_data.get("label", source),
                            "formula": source_data.get("formula", ""),
                            "affects": target_data.get("label", target),
                            "description": data.get("description", ""),
                            "strength": data.get("strength", 0.5),
                        }
                    )

        return sorted(results, key=lambda x: x.get("strength", 0), reverse=True)
