"""Visualization module for knowledge graph rendering with Plotly."""

import math
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import plotly.graph_objects as go

from .knowledge_graph import KnowledgeGraph


class Visualizer:
    """Creates interactive visualizations of knowledge graphs using Plotly."""
    
    # Modern color palette
    COLORS = {
        "Material": "#10B981",      # Emerald green
        "Component": "#3B82F6",     # Blue
        "Property": "#F59E0B",      # Amber
        "Source": "#8B5CF6",        # Purple
        "default": "#6B7280",       # Gray
    }
    
    EDGE_COLORS = {
        "CONTAINS": "#10B981",
        "AFFECTS": "#F59E0B",
        "REACTS_WITH": "#EF4444",
        "CITED_IN": "#8B5CF6",
        "PRODUCES": "#06B6D4",
        "default": "#9CA3AF",
    }
    
    def __init__(self, dark_mode: bool = True):
        """
        Initialize the visualizer.
        
        Args:
            dark_mode: Whether to use dark mode styling
        """
        self.dark_mode = dark_mode
        self.bg_color = "#0F172A" if dark_mode else "#FFFFFF"
        self.text_color = "#F8FAFC" if dark_mode else "#1E293B"
        self.grid_color = "#334155" if dark_mode else "#E2E8F0"
    
    def create_network_graph(
        self,
        kg: KnowledgeGraph,
        width: int = 800,
        height: int = 600,
        title: str = "Knowledge Graph",
        layout_algo: str = "spring"
    ) -> go.Figure:
        """
        Create an interactive network graph visualization.
        
        Args:
            kg: KnowledgeGraph instance
            width: Figure width in pixels
            height: Figure height in pixels
            title: Graph title
            layout_algo: Layout algorithm ('spring', 'circular', 'kamada_kawai')
            
        Returns:
            Plotly Figure object
        """
        if kg.graph.number_of_nodes() == 0:
            return self._create_empty_figure(width, height, "No data in knowledge graph")
        
        # Compute layout
        pos = self._compute_layout(kg.graph, layout_algo)
        
        # Create edge traces
        edge_traces = self._create_edge_traces(kg.graph, pos)
        
        # Create node traces
        node_trace = self._create_node_trace(kg.graph, pos)
        
        # Create figure
        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title=dict(
                    text=title,
                    font=dict(size=20, color=self.text_color),
                    x=0.5,
                    xanchor="center"
                ),
                showlegend=True,
                hovermode="closest",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                paper_bgcolor=self.bg_color,
                plot_bgcolor=self.bg_color,
                width=width,
                height=height,
                margin=dict(l=20, r=20, t=60, b=20),
                legend=dict(
                    font=dict(color=self.text_color),
                    bgcolor="rgba(0,0,0,0.3)" if self.dark_mode else "rgba(255,255,255,0.8)",
                    bordercolor=self.grid_color,
                    borderwidth=1
                ),
                annotations=self._create_edge_labels(kg.graph, pos)
            )
        )
        
        return fig
    
    def _compute_layout(self, graph: nx.DiGraph, algo: str) -> Dict[str, Tuple[float, float]]:
        """Compute node positions using specified layout algorithm."""
        if graph.number_of_nodes() == 0:
            return {}
        
        try:
            if algo == "circular":
                return nx.circular_layout(graph)
            elif algo == "kamada_kawai":
                return nx.kamada_kawai_layout(graph)
            else:  # spring is default
                return nx.spring_layout(graph, k=2/math.sqrt(graph.number_of_nodes()), iterations=50)
        except Exception:
            return nx.spring_layout(graph)
    
    def _create_edge_traces(
        self, 
        graph: nx.DiGraph, 
        pos: Dict[str, Tuple[float, float]]
    ) -> List[go.Scatter]:
        """Create edge traces for the graph."""
        traces = []
        edge_types = set()
        
        for source, target, data in graph.edges(data=True):
            if source not in pos or target not in pos:
                continue
            
            rel_type = data.get("relationship_type", "RELATED")
            edge_types.add(rel_type)
            
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            
            # Create edge line
            color = self.EDGE_COLORS.get(rel_type, self.EDGE_COLORS["default"])
            
            trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=1.5, color=color),
                hoverinfo="text",
                hovertext=f"{rel_type}: {data.get('description', '')}",
                showlegend=False,
                opacity=0.6
            )
            traces.append(trace)
        
        # Add legend entries for relationship types
        for rel_type in edge_types:
            color = self.EDGE_COLORS.get(rel_type, self.EDGE_COLORS["default"])
            traces.append(go.Scatter(
                x=[None], y=[None],
                mode="lines",
                line=dict(width=3, color=color),
                name=rel_type,
                showlegend=True
            ))
        
        return traces
    
    def _create_node_trace(
        self, 
        graph: nx.DiGraph, 
        pos: Dict[str, Tuple[float, float]]
    ) -> go.Scatter:
        """Create node trace for the graph."""
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        for node_id, data in graph.nodes(data=True):
            if node_id not in pos:
                continue
            
            x, y = pos[node_id]
            node_x.append(x)
            node_y.append(y)
            
            # Build hover text
            label = data.get("label", node_id)
            node_type = data.get("node_type", "Unknown")
            
            hover_parts = [f"<b>{label}</b>", f"Type: {node_type}"]
            
            if data.get("formula"):
                hover_parts.append(f"Formula: {data['formula']}")
            if data.get("percentage"):
                hover_parts.append(f"Percentage: {data['percentage']}%")
            if data.get("role"):
                hover_parts.append(f"Role: {data['role']}")
            
            node_text.append("<br>".join(hover_parts))
            
            # Set color based on type
            node_colors.append(self.COLORS.get(node_type, self.COLORS["default"]))
            
            # Set size based on degree
            degree = graph.degree(node_id)
            node_sizes.append(max(15, min(40, 10 + degree * 5)))
        
        return go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            hoverinfo="text",
            hovertext=node_text,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color=self.bg_color),
                opacity=0.9
            ),
            text=[data.get("label", "")[:15] for _, data in graph.nodes(data=True) if _ in pos],
            textposition="top center",
            textfont=dict(size=10, color=self.text_color),
            showlegend=False
        )
    
    def _create_edge_labels(
        self, 
        graph: nx.DiGraph, 
        pos: Dict[str, Tuple[float, float]]
    ) -> List[Dict]:
        """Create edge labels as annotations (optional, for key relationships)."""
        annotations = []
        
        # Only label important relationships
        for source, target, data in graph.edges(data=True):
            if source not in pos or target not in pos:
                continue
            
            strength = data.get("strength", 0)
            if strength >= 0.8:  # Only label strong relationships
                x0, y0 = pos[source]
                x1, y1 = pos[target]
                
                annotations.append(dict(
                    x=(x0 + x1) / 2,
                    y=(y0 + y1) / 2,
                    text=data.get("relationship_type", ""),
                    showarrow=False,
                    font=dict(size=8, color=self.text_color),
                    opacity=0.7
                ))
        
        return annotations
    
    def create_composition_sunburst(
        self,
        kg: KnowledgeGraph,
        material_name: Optional[str] = None,
        width: int = 600,
        height: int = 600
    ) -> go.Figure:
        """
        Create a sunburst chart showing material composition hierarchy.
        
        Args:
            kg: KnowledgeGraph instance
            material_name: Specific material to show (None for all)
            width: Figure width
            height: Figure height
            
        Returns:
            Plotly Figure object
        """
        ids = []
        labels = []
        parents = []
        values = []
        colors = []
        
        # Get materials
        materials = kg.get_nodes_by_type("Material")
        if material_name:
            materials = [m for m in materials if m.get("label") == material_name]
        
        if not materials:
            return self._create_empty_figure(width, height, "No materials found")
        
        # Root node
        ids.append("root")
        labels.append("Concrete Materials")
        parents.append("")
        values.append(100)
        colors.append(self.COLORS["default"])
        
        for mat in materials:
            mat_id = mat["id"]
            mat_label = mat.get("label", mat_id)
            
            ids.append(mat_id)
            labels.append(mat_label)
            parents.append("root")
            values.append(100)
            colors.append(self.COLORS["Material"])
            
            # Get components
            neighbors = kg.get_neighbors(mat_id)
            components = [n for n in neighbors if n.get("node_type") == "Component"]
            
            for comp in components:
                pct = comp.get("percentage", 10)
                comp_id = f"{mat_id}_{comp['id']}"
                
                ids.append(comp_id)
                labels.append(f"{comp.get('label', '')} ({comp.get('formula', '')})")
                parents.append(mat_id)
                values.append(pct if pct else 10)
                colors.append(self.COLORS["Component"])
        
        fig = go.Figure(go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(colors=colors),
            branchvalues="total",
            hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
        ))
        
        fig.update_layout(
            title=dict(
                text="Material Composition Hierarchy",
                font=dict(size=18, color=self.text_color),
                x=0.5
            ),
            paper_bgcolor=self.bg_color,
            width=width,
            height=height,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        return fig
    
    def create_component_heatmap(
        self,
        kg: KnowledgeGraph,
        width: int = 700,
        height: int = 500
    ) -> go.Figure:
        """
        Create a heatmap showing component relationships/interactions.
        
        Args:
            kg: KnowledgeGraph instance
            width: Figure width
            height: Figure height
            
        Returns:
            Plotly Figure object
        """
        # Get components
        components = kg.get_nodes_by_type("Component")
        
        if len(components) < 2:
            return self._create_empty_figure(width, height, "Not enough components for heatmap")
        
        # Build adjacency matrix
        comp_ids = [c["id"] for c in components]
        comp_labels = [f"{c.get('label', '')} ({c.get('formula', '')})" for c in components]
        
        n = len(comp_ids)
        matrix = [[0.0] * n for _ in range(n)]
        
        for i, src_id in enumerate(comp_ids):
            for j, tgt_id in enumerate(comp_ids):
                if i == j:
                    matrix[i][j] = 1.0
                elif kg.graph.has_edge(src_id, tgt_id):
                    edge_data = kg.graph.edges[src_id, tgt_id]
                    matrix[i][j] = edge_data.get("strength", 0.5)
                elif kg.graph.has_edge(tgt_id, src_id):
                    edge_data = kg.graph.edges[tgt_id, src_id]
                    matrix[i][j] = edge_data.get("strength", 0.5)
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=comp_labels,
            y=comp_labels,
            colorscale="Viridis",
            hovertemplate="From: %{y}<br>To: %{x}<br>Strength: %{z:.2f}<extra></extra>"
        ))
        
        fig.update_layout(
            title=dict(
                text="Component Interaction Matrix",
                font=dict(size=18, color=self.text_color),
                x=0.5
            ),
            paper_bgcolor=self.bg_color,
            plot_bgcolor=self.bg_color,
            width=width,
            height=height,
            xaxis=dict(
                tickangle=45,
                tickfont=dict(color=self.text_color, size=10),
                side="bottom"
            ),
            yaxis=dict(
                tickfont=dict(color=self.text_color, size=10),
                autorange="reversed"
            ),
            margin=dict(l=150, r=20, t=50, b=150)
        )
        
        return fig
    
    def create_statistics_cards(self, kg: KnowledgeGraph) -> Dict[str, Any]:
        """
        Generate statistics data for display cards.
        
        Args:
            kg: KnowledgeGraph instance
            
        Returns:
            Dictionary of statistics
        """
        stats = kg.get_stats()
        
        return {
            "total_nodes": stats["total_nodes"],
            "total_edges": stats["total_edges"],
            "materials": stats["node_types"].get("Material", 0),
            "components": stats["node_types"].get("Component", 0),
            "properties": stats["node_types"].get("Property", 0),
            "sources": stats["node_types"].get("Source", 0),
            "density": round(stats["density"] * 100, 2),
            "relationship_breakdown": stats["edge_types"]
        }
    
    def _create_empty_figure(
        self, 
        width: int, 
        height: int, 
        message: str
    ) -> go.Figure:
        """Create an empty figure with a message."""
        fig = go.Figure()
        
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=self.text_color)
        )
        
        fig.update_layout(
            paper_bgcolor=self.bg_color,
            plot_bgcolor=self.bg_color,
            width=width,
            height=height,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        
        return fig
