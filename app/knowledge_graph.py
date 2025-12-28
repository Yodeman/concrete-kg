"""Knowledge Graph engine with NetworkX and Neo4j support."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from .config import get_config
from .llm_extractor import (
    ChemicalComponent,
    ComponentRelationship,
    ExtractionResult,
    MaterialComposition,
)


@dataclass
class NodeData:
    """Represents a node in the knowledge graph."""
    id: str
    label: str
    node_type: str  # Material, Component, Property, Source
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class EdgeData:
    """Represents an edge in the knowledge graph."""
    source: str
    target: str
    relationship_type: str  # CONTAINS, AFFECTS, REACTS_WITH, CITED_IN
    properties: Dict[str, Any] = field(default_factory=dict)


class KnowledgeGraph:
    """
    Knowledge graph for concrete material compositions.
    
    Supports dual storage: NetworkX (in-memory) and Neo4j (persistent).
    """
    
    # Node type colors for visualization
    NODE_COLORS = {
        "Material": "#4CAF50",      # Green
        "Component": "#2196F3",     # Blue
        "Property": "#FF9800",      # Orange
        "Source": "#9C27B0",        # Purple
    }
    
    # Relationship type styles
    EDGE_STYLES = {
        "CONTAINS": {"color": "#4CAF50", "weight": 3},
        "AFFECTS": {"color": "#FF9800", "weight": 2},
        "REACTS_WITH": {"color": "#F44336", "weight": 2},
        "CITED_IN": {"color": "#9C27B0", "weight": 1},
        "PRODUCES": {"color": "#00BCD4", "weight": 2},
    }
    
    def __init__(self, use_neo4j: bool = False):
        """
        Initialize the knowledge graph.
        
        Args:
            use_neo4j: Whether to sync with Neo4j database
        """
        self.graph = nx.DiGraph()
        self.use_neo4j = use_neo4j
        self._neo4j_graph = None
        
        if use_neo4j:
            self._init_neo4j()
    
    def _init_neo4j(self):
        """Initialize Neo4j connection."""
        try:
            from langchain_neo4j import Neo4jGraph
            config = get_config()
            
            if config.neo4j.is_configured:
                self._neo4j_graph = Neo4jGraph(
                    url=config.neo4j.uri,
                    username=config.neo4j.username,
                    password=config.neo4j.password,
                )
                print("Connected to Neo4j successfully")
            else:
                print("Neo4j not configured, using in-memory graph only")
                self.use_neo4j = False
        except ImportError:
            print("langchain-neo4j not installed, using in-memory graph only")
            self.use_neo4j = False
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            self.use_neo4j = False
    
    @property
    def neo4j_graph(self):
        """Get Neo4j graph instance."""
        return self._neo4j_graph
    
    # =========================================================================
    # Node Operations
    # =========================================================================
    
    def add_node(self, node_id: str, label: str, node_type: str, 
                 properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a node to the graph.
        
        Args:
            node_id: Unique identifier for the node
            label: Display label
            node_type: Type (Material, Component, Property, Source)
            properties: Additional properties
            
        Returns:
            Node ID
        """
        props = properties or {}
        self.graph.add_node(
            node_id,
            label=label,
            node_type=node_type,
            color=self.NODE_COLORS.get(node_type, "#757575"),
            **props
        )
        return node_id
    
    def add_material(self, name: str, properties: Optional[Dict[str, Any]] = None) -> str:
        """Add a material node."""
        node_id = f"material:{name.lower().replace(' ', '_')}"
        return self.add_node(node_id, name, "Material", properties)
    
    def add_component(self, name: str, formula: str, role: str,
                      percentage: Optional[float] = None) -> str:
        """Add a chemical component node."""
        node_id = f"component:{formula}"
        props = {"formula": formula, "role": role}
        if percentage is not None:
            props["percentage"] = percentage
        return self.add_node(node_id, name, "Component", props)
    
    def add_property(self, name: str, value: Optional[str] = None) -> str:
        """Add a property node."""
        node_id = f"property:{name.lower().replace(' ', '_')}"
        props = {}
        if value:
            props["value"] = value
        return self.add_node(node_id, name, "Property", props)
    
    def add_source(self, title: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a source document node."""
        node_id = f"source:{title.lower().replace(' ', '_')[:50]}"
        return self.add_node(node_id, title, "Source", metadata)
    
    # =========================================================================
    # Edge Operations
    # =========================================================================
    
    def add_edge(self, source: str, target: str, relationship_type: str,
                 properties: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """
        Add an edge to the graph.
        
        Args:
            source: Source node ID
            target: Target node ID
            relationship_type: Type of relationship
            properties: Additional properties
            
        Returns:
            Tuple of (source, target)
        """
        props = properties or {}
        style = self.EDGE_STYLES.get(relationship_type, {"color": "#757575", "weight": 1})
        
        self.graph.add_edge(
            source, target,
            relationship_type=relationship_type,
            color=style["color"],
            weight=style["weight"],
            **props
        )
        return (source, target)
    
    def add_relationship(self, source_label: str, target_label: str, 
                         relationship_type: str, 
                         properties: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """
        Add a relationship, resolving labels to node IDs.
        
        Args:
            source_label: Source node label or formula
            target_label: Target node label or formula
            relationship_type: Type of relationship
            properties: Additional properties
        """
        source_id = self._resolve_node_id(source_label)
        target_id = self._resolve_node_id(target_label)
        
        return self.add_edge(source_id, target_id, relationship_type, properties)
    
    def _resolve_node_id(self, label: str) -> str:
        """Resolve a label to a node ID, or create one."""
        # Check if it's already an ID in the graph
        if label in self.graph.nodes:
            return label
        
        # Try to find by label
        for node_id, data in self.graph.nodes(data=True):
            if data.get("label") == label:
                return node_id
            # Also check formula for components
            if data.get("formula") == label:
                return node_id
        
        # Not found, create a generic component
        return f"component:{label}"
    
    # =========================================================================
    # Build from Extraction
    # =========================================================================
    
    def build_from_extraction(self, result: ExtractionResult) -> "KnowledgeGraph":
        """
        Build the knowledge graph from an extraction result.
        
        Args:
            result: ExtractionResult from LLM extraction
            
        Returns:
            Self for chaining
        """
        # Add source if available
        if result.source_document:
            source_id = self.add_source(result.source_document)
        else:
            source_id = None
        
        # Add materials and their components
        for material in result.materials:
            if isinstance(material, dict):
                material = MaterialComposition(**material)
            
            material_id = self.add_material(
                material.material_name,
                {"type": material.material_type, **material.properties}
            )
            
            # Link to source
            if source_id:
                self.add_edge(material_id, source_id, "CITED_IN")
            
            # Add components
            for comp in material.components:
                if isinstance(comp, dict):
                    comp = ChemicalComponent(**comp)
                
                component_id = self.add_component(
                    comp.name, 
                    comp.formula, 
                    comp.role,
                    comp.percentage
                )
                
                # Link material to component
                props = {}
                if comp.percentage:
                    props["percentage"] = comp.percentage
                self.add_edge(material_id, component_id, "CONTAINS", props)
        
        # Add explicit relationships
        for rel in result.relationships:
            if isinstance(rel, dict):
                rel = ComponentRelationship(**rel)
            
            # Ensure nodes exist
            source_node = self._resolve_node_id(rel.source)
            target_node = self._resolve_node_id(rel.target)
            
            # Add nodes if they don't exist
            if source_node not in self.graph.nodes:
                self.add_node(source_node, rel.source, "Component", {})
            if target_node not in self.graph.nodes:
                # Check if it looks like a property
                if any(p in rel.target.lower() for p in ["strength", "durability", "time", "color"]):
                    self.add_property(rel.target)
                else:
                    self.add_node(target_node, rel.target, "Component", {})
            
            props = {"description": rel.description}
            if rel.strength:
                props["strength"] = rel.strength
            
            self.add_edge(source_node, target_node, rel.relationship_type, props)
        
        return self
    
    # =========================================================================
    # Neo4j Sync
    # =========================================================================
    
    def sync_to_neo4j(self) -> bool:
        """
        Sync the in-memory graph to Neo4j.
        
        Returns:
            True if successful
        """
        if not self.use_neo4j or not self._neo4j_graph:
            return False
        
        try:
            # Clear existing data (optional, for clean sync)
            # self._neo4j_graph.query("MATCH (n) DETACH DELETE n")
            
            # Create nodes
            for node_id, data in self.graph.nodes(data=True):
                node_type = data.get("node_type", "Node")
                props = {k: v for k, v in data.items() if k != "node_type"}
                props["id"] = node_id
                
                query = f"""
                MERGE (n:{node_type} {{id: $id}})
                SET n += $props
                """
                self._neo4j_graph.query(query, {"id": node_id, "props": props})
            
            # Create relationships
            for source, target, data in self.graph.edges(data=True):
                rel_type = data.get("relationship_type", "RELATES_TO")
                props = {k: v for k, v in data.items() if k != "relationship_type"}
                
                query = f"""
                MATCH (s {{id: $source}})
                MATCH (t {{id: $target}})
                MERGE (s)-[r:{rel_type}]->(t)
                SET r += $props
                """
                self._neo4j_graph.query(query, {
                    "source": source,
                    "target": target,
                    "props": props
                })
            
            print(f"Synced {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges to Neo4j")
            return True
            
        except Exception as e:
            print(f"Error syncing to Neo4j: {e}")
            return False
    
    def load_from_neo4j(self) -> bool:
        """
        Load graph from Neo4j into memory.
        
        Returns:
            True if successful
        """
        if not self.use_neo4j or not self._neo4j_graph:
            return False
        
        try:
            # Clear in-memory graph
            self.graph.clear()
            
            # Load nodes
            result = self._neo4j_graph.query("""
                MATCH (n)
                RETURN n.id as id, labels(n)[0] as type, properties(n) as props
            """)
            
            for record in result:
                node_id = record["id"]
                node_type = record["type"]
                props = record["props"]
                label = props.pop("label", node_id)
                self.add_node(node_id, label, node_type, props)
            
            # Load relationships
            result = self._neo4j_graph.query("""
                MATCH (s)-[r]->(t)
                RETURN s.id as source, t.id as target, type(r) as rel_type, properties(r) as props
            """)
            
            for record in result:
                self.add_edge(
                    record["source"],
                    record["target"],
                    record["rel_type"],
                    record["props"]
                )
            
            print(f"Loaded {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges from Neo4j")
            return True
            
        except Exception as e:
            print(f"Error loading from Neo4j: {e}")
            return False
    
    # =========================================================================
    # Query Operations
    # =========================================================================
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by ID."""
        if node_id in self.graph.nodes:
            return dict(self.graph.nodes[node_id])
        return None
    
    def get_neighbors(self, node_id: str) -> List[Dict[str, Any]]:
        """Get all neighbors of a node."""
        if node_id not in self.graph.nodes:
            return []
        
        neighbors = []
        for neighbor_id in self.graph.neighbors(node_id):
            data = dict(self.graph.nodes[neighbor_id])
            data["id"] = neighbor_id
            edge_data = self.graph.edges[node_id, neighbor_id]
            data["relationship"] = edge_data.get("relationship_type", "RELATED")
            neighbors.append(data)
        
        return neighbors
    
    def get_nodes_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        """Get all nodes of a specific type."""
        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("node_type") == node_type:
                node_data = dict(data)
                node_data["id"] = node_id
                nodes.append(node_data)
        return nodes
    
    def get_subgraph(self, center_node: str, depth: int = 2) -> nx.DiGraph:
        """Get a subgraph around a center node."""
        if center_node not in self.graph.nodes:
            return nx.DiGraph()
        
        # BFS to find nodes within depth
        nodes_to_include = {center_node}
        current_level = {center_node}
        
        for _ in range(depth):
            next_level = set()
            for node in current_level:
                next_level.update(self.graph.neighbors(node))
                next_level.update(self.graph.predecessors(node))
            nodes_to_include.update(next_level)
            current_level = next_level
        
        return self.graph.subgraph(nodes_to_include).copy()
    
    def query_cypher(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """
        Execute a Cypher query on Neo4j.
        
        Args:
            query: Cypher query string
            params: Query parameters
            
        Returns:
            Query results
        """
        if not self.use_neo4j or not self._neo4j_graph:
            raise RuntimeError("Neo4j not configured")
        
        return self._neo4j_graph.query(query, params or {})
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary for serialization."""
        return {
            "nodes": [
                {"id": node_id, **data}
                for node_id, data in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": s, "target": t, **data}
                for s, t, data in self.graph.edges(data=True)
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], use_neo4j: bool = False) -> "KnowledgeGraph":
        """Create graph from dictionary."""
        kg = cls(use_neo4j=use_neo4j)
        
        for node in data.get("nodes", []):
            node_id = node.pop("id")
            label = node.pop("label", node_id)
            node_type = node.pop("node_type", "Node")
            node.pop("color", None)  # Remove computed property
            kg.add_node(node_id, label, node_type, node)
        
        for edge in data.get("edges", []):
            source = edge.pop("source")
            target = edge.pop("target")
            rel_type = edge.pop("relationship_type", "RELATES_TO")
            edge.pop("color", None)
            edge.pop("weight", None)
            kg.add_edge(source, target, rel_type, edge)
        
        return kg
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2)
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        node_types = {}
        for _, data in self.graph.nodes(data=True):
            nt = data.get("node_type", "Unknown")
            node_types[nt] = node_types.get(nt, 0) + 1
        
        edge_types = {}
        for _, _, data in self.graph.edges(data=True):
            et = data.get("relationship_type", "Unknown")
            edge_types[et] = edge_types.get(et, 0) + 1
        
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": node_types,
            "edge_types": edge_types,
            "density": nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0,
        }
