"""
Concrete Knowledge Graph Application - Streamlit Web Interface

A tool for extracting chemical compositions from research papers
and visualizing them as interactive knowledge graphs.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path for imports when running via streamlit
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st

from app.config import get_config, reload_config
from app.graph_rag import GraphRAG
from app.knowledge_graph import KnowledgeGraph
from app.llm_extractor import LLMExtractor, get_demo_extraction_result
from app.pdf_processor import PDFProcessor
from app.visualizer import Visualizer
from app.reasoning.chains import GraphReasoningChain
from app.reasoning.hypothesis import HypothesisGenerator


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Concrete Knowledge Graph",
    page_icon="🧱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #475569;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .stat-value {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #10B981 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label {
        color: #94A3B8;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .chat-message {
        background: #1E293B;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #3B82F6;
    }
    .user-message {
        border-left-color: #10B981;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "knowledge_graph" not in st.session_state:
        st.session_state.knowledge_graph = KnowledgeGraph(use_neo4j=False)
    if "extraction_result" not in st.session_state:
        st.session_state.extraction_result = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "demo_loaded" not in st.session_state:
        st.session_state.demo_loaded = False
    if "visualizer" not in st.session_state:
        st.session_state.visualizer = Visualizer(dark_mode=True)


init_session_state()


# =============================================================================
# Sidebar
# =============================================================================

def render_sidebar():
    """Render the sidebar with configuration and upload options."""
    with st.sidebar:
        st.title("🧱 Concrete KG")
        st.markdown("---")
        
        # Demo Mode Toggle
        st.subheader("🎮 Quick Start")
        if st.button("Load Demo Data", use_container_width=True, type="primary"):
            load_demo_data()
        
        st.markdown("---")
        
        # PDF Upload
        st.subheader("📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Upload Research Paper (PDF)",
            type=["pdf"],
            help="Upload a PDF document about concrete materials"
        )
        
        if uploaded_file:
            st.success(f"Uploaded: {uploaded_file.name}")
            if st.button("Extract Knowledge", use_container_width=True):
                extract_from_pdf(uploaded_file)
        
        st.markdown("---")
        
        # Configuration
        st.subheader("⚙️ Configuration")
        
        # LLM Configuration
        with st.expander("LLM Settings"):
            config = get_config()
            
            api_key = st.text_input(
                "OpenAI API Key",
                value=config.llm.api_key if config.llm.is_configured else "",
                type="password",
                help="Your OpenAI API key for LLM extraction"
            )
            
            if api_key and api_key != config.llm.api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                reload_config()
                st.success("API key updated!")
            
            model = st.selectbox(
                "Model",
                ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
                index=0
            )
            if model != config.llm.model:
                os.environ["LLM_MODEL"] = model
                reload_config()
        
        # Neo4j Configuration
        with st.expander("Neo4j Settings"):
            config = get_config()
            
            neo4j_uri = st.text_input(
                "Neo4j URI",
                value=config.neo4j.uri if config.neo4j.is_configured else "",
                placeholder="neo4j+s://xxx.databases.neo4j.io"
            )
            neo4j_user = st.text_input(
                "Username",
                value=config.neo4j.username
            )
            neo4j_pass = st.text_input(
                "Password",
                type="password"
            )
            
            use_neo4j = st.checkbox("Enable Neo4j Sync", value=False)
            
            if st.button("Connect to Neo4j"):
                if neo4j_uri and neo4j_pass:
                    os.environ["NEO4J_URI"] = neo4j_uri
                    os.environ["NEO4J_USERNAME"] = neo4j_user
                    os.environ["NEO4J_PASSWORD"] = neo4j_pass
                    reload_config()
                    
                    # Reinitialize graph with Neo4j
                    st.session_state.knowledge_graph = KnowledgeGraph(use_neo4j=True)
                    if st.session_state.knowledge_graph.use_neo4j:
                        st.success("Connected to Neo4j!")
                    else:
                        st.error("Failed to connect to Neo4j")
        
        st.markdown("---")
        
        # Graph Statistics
        if st.session_state.knowledge_graph.graph.number_of_nodes() > 0:
            st.subheader("📊 Graph Stats")
            stats = st.session_state.knowledge_graph.get_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Nodes", stats["total_nodes"])
                st.metric("Materials", stats["node_types"].get("Material", 0))
            with col2:
                st.metric("Edges", stats["total_edges"])
                st.metric("Components", stats["node_types"].get("Component", 0))


def load_demo_data():
    """Load demo data into the knowledge graph."""
    with st.spinner("Loading demo data..."):
        # Get demo extraction result
        result = get_demo_extraction_result()
        st.session_state.extraction_result = result
        
        # Build knowledge graph
        kg = KnowledgeGraph(use_neo4j=st.session_state.knowledge_graph.use_neo4j)
        kg.build_from_extraction(result)
        st.session_state.knowledge_graph = kg
        st.session_state.demo_loaded = True
        
    st.success("Demo data loaded successfully!")
    st.rerun()


def extract_from_pdf(uploaded_file):
    """Extract knowledge from uploaded PDF."""
    config = get_config()
    
    if not config.llm.is_configured:
        st.error("Please configure your OpenAI API key first.")
        return
    
    with st.spinner("Extracting knowledge from PDF... This may take a minute."):
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            
            # Extract using LLM
            extractor = LLMExtractor()
            result = extractor.process_document(tmp_path)
            st.session_state.extraction_result = result
            
            # Build knowledge graph
            kg = KnowledgeGraph(use_neo4j=st.session_state.knowledge_graph.use_neo4j)
            kg.build_from_extraction(result)
            st.session_state.knowledge_graph = kg
            
            # Sync to Neo4j if enabled
            if kg.use_neo4j:
                kg.sync_to_neo4j()
            
            # Cleanup
            os.unlink(tmp_path)
            
            st.success(f"Extracted {len(result.materials)} materials and {len(result.relationships)} relationships!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Extraction failed: {str(e)}")


# =============================================================================
# Main Content Tabs
# =============================================================================

def render_main_content():
    """Render the main content area with tabs."""
    st.title("🧱 Concrete Materials Knowledge Graph")
    st.markdown("Extract and visualize chemical compositions from research papers")
    
    # Check if we have data
    has_data = st.session_state.knowledge_graph.graph.number_of_nodes() > 0
    
    if not has_data:
        render_welcome_screen()
        return
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Extraction Results",
        "🕸️ Knowledge Graph",
        "🔬 Laboratory",
        "💬 Graph RAG Q&A",
        "📥 Export"
    ])
    
    with tab1:
        render_extraction_results()
    
    with tab2:
        render_knowledge_graph()
        
    with tab3:
        render_laboratory()
    
    with tab4:
        render_graph_rag()
    
    with tab5:
        render_export()


def render_welcome_screen():
    """Render welcome screen when no data is loaded."""
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">1</div>
            <div class="stat-label">Upload PDF</div>
            <p style="color: #CBD5E1; margin-top: 0.5rem;">
                Upload a research paper about concrete materials
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">2</div>
            <div class="stat-label">Extract Knowledge</div>
            <p style="color: #CBD5E1; margin-top: 0.5rem;">
                AI extracts compositions and relationships
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">3</div>
            <div class="stat-label">Explore Graph</div>
            <p style="color: #CBD5E1; margin-top: 0.5rem;">
                Interactive visualization and Q&A
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("👈 Click **Load Demo Data** in the sidebar to get started, or upload a PDF!")


def render_extraction_results():
    """Render extraction results tab."""
    result = st.session_state.extraction_result
    
    if not result:
        st.info("No extraction results available.")
        return
    
    # Statistics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📦 Materials", len(result.materials))
    with col2:
        total_components = sum(len(m.components) for m in result.materials)
        st.metric("🧪 Components", total_components)
    with col3:
        st.metric("🔗 Relationships", len(result.relationships))
    with col4:
        st.metric("📄 Source", result.source_document[:30] + "..." if len(result.source_document) > 30 else result.source_document)
    
    st.markdown("---")
    
    # Materials table
    st.subheader("📦 Extracted Materials")
    
    for i, material in enumerate(result.materials):
        with st.expander(f"**{material.material_name}** ({material.material_type})", expanded=(i == 0)):
            # Properties
            if material.properties:
                st.markdown("**Properties:**")
                for key, value in material.properties.items():
                    st.markdown(f"- {key}: {value}")
            
            # Components table
            if material.components:
                st.markdown("**Chemical Composition:**")
                
                import pandas as pd
                df = pd.DataFrame([
                    {
                        "Name": c.name,
                        "Formula": c.formula,
                        "Percentage (%)": c.percentage if c.percentage else "-",
                        "Role": c.role
                    }
                    for c in material.components
                ])
                st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Relationships
    st.markdown("---")
    st.subheader("🔗 Extracted Relationships")
    
    import pandas as pd
    if result.relationships:
        rel_df = pd.DataFrame([
            {
                "Source": r.source,
                "Relationship": r.relationship_type,
                "Target": r.target,
                "Description": r.description[:50] + "..." if len(r.description) > 50 else r.description
            }
            for r in result.relationships
        ])
        st.dataframe(rel_df, use_container_width=True, hide_index=True)
    else:
        st.info("No relationships extracted.")


def render_knowledge_graph():
    """Render knowledge graph visualization tab."""
    kg = st.session_state.knowledge_graph
    viz = st.session_state.visualizer
    
    # Visualization options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        viz_type = st.selectbox(
            "Visualization Type",
            ["Network Graph", "Composition Sunburst", "Interaction Heatmap"]
        )
    
    with col2:
        if viz_type == "Network Graph":
            layout = st.selectbox("Layout", ["spring", "circular", "kamada_kawai"])
        else:
            layout = "spring"
    
    with col3:
        if viz_type == "Composition Sunburst":
            materials = kg.get_nodes_by_type("Material")
            material_names = ["All Materials"] + [m["label"] for m in materials]
            selected_material = st.selectbox("Material", material_names)
        else:
            selected_material = None
    
    # Render visualization
    st.markdown("---")
    
    if viz_type == "Network Graph":
        fig = viz.create_network_graph(kg, width=1000, height=700, layout_algo=layout)
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Composition Sunburst":
        mat_name = None if selected_material == "All Materials" else selected_material
        fig = viz.create_composition_sunburst(kg, material_name=mat_name, width=700, height=700)
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Interaction Heatmap":
        fig = viz.create_component_heatmap(kg, width=900, height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    # Node explorer
    st.markdown("---")
    st.subheader("🔍 Node Explorer")
    
    nodes = list(kg.graph.nodes(data=True))
    if nodes:
        node_options = {f"{data.get('label', nid)} ({data.get('node_type', 'Unknown')})": nid 
                       for nid, data in nodes}
        selected_node_label = st.selectbox("Select Node", list(node_options.keys()))
        selected_node = node_options[selected_node_label]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Node Properties:**")
            node_data = kg.get_node(selected_node)
            if node_data:
                for key, value in node_data.items():
                    if key != "color":
                        st.markdown(f"- **{key}:** {value}")
        
        with col2:
            st.markdown("**Connected Nodes:**")
            neighbors = kg.get_neighbors(selected_node)
            for neighbor in neighbors:
                rel = neighbor.get("relationship", "RELATED")
                st.markdown(f"- {rel} → {neighbor.get('label', neighbor.get('id', 'Unknown'))}")


def render_graph_rag():
    """Render Graph RAG Q&A tab."""
    kg = st.session_state.knowledge_graph
    
    st.subheader("💬 Ask Questions About the Knowledge Graph")
    st.markdown("Use natural language to query the extracted knowledge.")
    
    # Initialize Graph RAG
    graph_rag = GraphRAG(kg)
    
    # Check configuration
    config = get_config()
    if not config.llm.is_configured:
        st.warning("⚠️ LLM not configured. Using keyword-based search only. Configure API key for better answers.")
    
    # Chat input
    question = st.text_input(
        "Ask a question",
        placeholder="e.g., What is the typical CaO percentage in Portland cement?",
        key="rag_question"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("Ask", type="primary", use_container_width=True)
    with col2:
        if st.button("Clear History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Process question
    if ask_button and question:
        with st.spinner("Thinking..."):
            result = graph_rag.ask(question)
            
            st.session_state.chat_history.append({
                "question": question,
                "answer": result["answer"],
                "method": result.get("method", "unknown")
            })
    
    # Display chat history
    st.markdown("---")
    
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong> {chat['question']}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="chat-message">
            <strong>Assistant:</strong> {chat['answer']}
            <br><small style="color: #64748B;">Method: {chat['method']}</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick queries
    st.markdown("---")
    st.subheader("🚀 Quick Queries")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Show all materials", use_container_width=True):
            materials = kg.get_nodes_by_type("Material")
            st.write([m["label"] for m in materials])
    
    with col2:
        if st.button("Show all components", use_container_width=True):
            components = kg.get_nodes_by_type("Component")
            st.write([f"{c['label']} ({c.get('formula', '')})" for c in components])
    
    with col3:
        if st.button("Component affecting strength", use_container_width=True):
            results = graph_rag.search_by_property("strength")
            if results:
                for r in results:
                    st.write(f"- {r['component']} ({r['formula']}): {r['description'][:50]}...")
            else:
                st.write("No results found")


def render_laboratory():
    """Render Laboratory tab for reasoning and hypothesis."""
    kg = st.session_state.knowledge_graph
    
    st.subheader("🔬 Concrete Laboratory")
    st.markdown("Use AI reasoning to analyze materials and generate new hypotheses.")
    
    # Mode selection
    mode = st.radio(
        "Select Mode",
        ["Reasoning Experiment", "Hypothesis Generator"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if mode == "Reasoning Experiment":
        st.markdown("### 🧠 Chain of Thought Reasoning")
        st.markdown("Ask complex questions that require multi-step logical deduction.")
        
        question = st.text_input(
            "Research Question",
            placeholder="e.g., How can we improve sulfate resistance without compromising early strength?",
            key="reasoning_q"
        )
        
        if st.button("Analyze", type="primary"):
            if not question:
                st.warning("Please enter a question.")
                return
                
            chain = GraphReasoningChain()
            
            with st.spinner("Reasoning through the problem..."):
                # Simple context retrieval (can be enhanced)
                nodes = kg.graph.nodes(data=True)
                context = str([
                    f"{d.get('label', n)} ({d.get('node_type')})" 
                    for n, d in nodes
                ])
                relationships = kg.graph.edges(data=True)
                context += "\nRelationships: " + str([
                    f"{u} {d.get('relationship_type')} {v}"
                    for u, v, d in relationships
                ])
                
                steps = chain.run_reasoning_chain(question, context)
                
                if steps:
                    st.success("Reasoning complete!")
                    
                    for step in steps:
                        with st.chat_message("assistant"):
                            st.markdown(f"**Step {step.step_number}:** {step.thought}")
                            st.info(f"💡 Conclusion: {step.conclusion}")
                else:
                    st.error("Failed to generate reasoning chain.")
                    
    elif mode == "Hypothesis Generator":
        st.markdown("### 🧪 Hypothesis Generator")
        st.markdown("Generate novel material compositions based on targets and constraints.")
        
        col1, col2 = st.columns(2)
        with col1:
            targets = st.multiselect(
                "Target Properties",
                ["High Strength", "High Durability", "Low Carbon/Green", "Fast Setting", "Sulfate Resistance", "Workability"],
                default=["High Durability"]
            )
        with col2:
            constraints = st.multiselect(
                "Constraints",
                ["No Silica Fume", "High Fly Ash Content", "Low Alkali", "No Superplasticizer"],
                default=[]
            )
            
        if st.button("Generate Hypothesis", type="primary"):
            gen = HypothesisGenerator()
            
            with st.spinner("Formulating hypothesis..."):
                # Context retrieval
                context = "Knowledge Graph Data:\n"
                for u, v, d in kg.graph.edges(data=True):
                    context += f"{u} --[{d.get('relationship_type')}]--> {v} ({d})\n"
                
                hypothesis = gen.generate_hypothesis(targets, constraints, context)
                
                if hypothesis:
                    st.success(f"Hypothesis Generated: {hypothesis.title}")
                    
                    st.markdown(f"**Description:** {hypothesis.description}")
                    
                    # Confidence Score
                    st.progress(hypothesis.confidence_score, text=f"Confidence Score: {hypothesis.confidence_score:.2f}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### 🧪 Proposed Composition")
                        # Display composition
                        comp_data = [
                            {"Component": c.name, "Percentage": c.percentage, "Role": c.role}
                            for c in hypothesis.proposed_composition.components
                        ]
                        st.dataframe(comp_data, hide_index=True)
                        
                    with col2:
                        st.markdown("#### 📊 Predicted Properties")
                        prop_data = [
                            {"Property": p.name, "Value": f"{p.value} {p.unit}"}
                            for p in hypothesis.predicted_properties
                        ]
                        st.dataframe(prop_data, hide_index=True)
                    
                    with st.expander("Show Reasoning Chain"):
                        for step in hypothesis.reasoning_chain:
                            st.markdown(f"**{step.step_number}.** {step.thought} → *{step.conclusion}*")
                            
                    if hypothesis.contradictions:
                        st.warning("⚠️ Potential Contradictions:")
                        for contra in hypothesis.contradictions:
                            st.markdown(f"- {contra}")
                            
                    # Add generic visualization of hypothesis node
                    if st.button("Add to Knowledge Graph"):
                        kg.add_node(
                            f"hypothesis:{hypothesis.title.lower().replace(' ', '_')}",
                            hypothesis.title,
                            "Hypothesis",
                            {"description": hypothesis.description}
                        )
                        st.success("Added hypothesis to graph!")
                else:
                    st.error("Failed to generate hypothesis.")


def render_export():
    """Render export tab."""
    kg = st.session_state.knowledge_graph
    
    st.subheader("📥 Export Knowledge Graph")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### JSON Export")
        st.markdown("Export the complete graph as JSON for further processing.")
        
        json_data = kg.to_json()
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name="concrete_knowledge_graph.json",
            mime="application/json",
            use_container_width=True
        )
        
        with st.expander("Preview JSON"):
            st.json(json.loads(json_data))
    
    with col2:
        st.markdown("### Neo4j Cypher Export")
        st.markdown("Generate Cypher statements to import into Neo4j.")
        
        cypher_statements = generate_cypher_export(kg)
        st.download_button(
            label="Download Cypher",
            data=cypher_statements,
            file_name="concrete_kg_import.cypher",
            mime="text/plain",
            use_container_width=True
        )
        
        with st.expander("Preview Cypher"):
            st.code(cypher_statements[:2000] + "..." if len(cypher_statements) > 2000 else cypher_statements, language="cypher")
    
    # Sync to Neo4j
    st.markdown("---")
    st.subheader("☁️ Sync to Neo4j AuraDB")
    
    if kg.use_neo4j and kg.neo4j_graph:
        st.success("✅ Connected to Neo4j")
        if st.button("Sync Graph to Neo4j", type="primary"):
            with st.spinner("Syncing..."):
                success = kg.sync_to_neo4j()
                if success:
                    st.success("Graph synced successfully!")
                else:
                    st.error("Sync failed. Check connection settings.")
    else:
        st.info("Configure Neo4j connection in the sidebar to enable cloud sync.")


def generate_cypher_export(kg: KnowledgeGraph) -> str:
    """Generate Cypher statements for Neo4j import."""
    lines = ["// Concrete Knowledge Graph - Neo4j Import Script", "// Generated by Concrete KG App", ""]
    
    # Create constraints
    lines.append("// Create constraints")
    for node_type in ["Material", "Component", "Property", "Source"]:
        lines.append(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{node_type}) REQUIRE n.id IS UNIQUE;")
    lines.append("")
    
    # Create nodes
    lines.append("// Create nodes")
    for node_id, data in kg.graph.nodes(data=True):
        node_type = data.get("node_type", "Node")
        props = {k: v for k, v in data.items() if k not in ["node_type", "color"]}
        props["id"] = node_id
        
        props_str = ", ".join(f'{k}: "{v}"' if isinstance(v, str) else f'{k}: {v}' 
                              for k, v in props.items())
        lines.append(f"MERGE (n:{node_type} {{{props_str}}});")
    lines.append("")
    
    # Create relationships
    lines.append("// Create relationships")
    for source, target, data in kg.graph.edges(data=True):
        rel_type = data.get("relationship_type", "RELATES_TO")
        props = {k: v for k, v in data.items() if k not in ["relationship_type", "color", "weight"]}
        
        props_str = ""
        if props:
            props_items = ", ".join(f'{k}: "{v}"' if isinstance(v, str) else f'{k}: {v}' 
                                   for k, v in props.items())
            props_str = f" {{{props_items}}}"
        
        lines.append(f'MATCH (s {{id: "{source}"}}), (t {{id: "{target}"}}) MERGE (s)-[r:{rel_type}{props_str}]->(t);')
    
    return "\n".join(lines)


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()
