# 🧱 Concrete Materials Knowledge Graph

A Python application for extracting and visualizing chemical compositions of concrete materials from scientific research papers using knowledge graph technology.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)
![Neo4j](https://img.shields.io/badge/Neo4j-AuraDB-008CC1.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-FF4B4B.svg)

## ✨ Features

- **📄 PDF Processing**: Extract text and tables from research papers using PyMuPDF
- **🤖 LLM Extraction**: Use LangChain + OpenAI to extract structured chemical compositions
- **🕸️ Knowledge Graph**: Build interactive graphs with NetworkX and Neo4j
- **💬 Graph RAG**: Natural language Q&A over the knowledge graph
- **📊 Visualization**: Interactive Plotly charts (network graphs, sunbursts, heatmaps)
- **☁️ Neo4j AuraDB**: Cloud persistence with Neo4j graph database

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Activate the virtual environment
pyenv activate concrete-kg

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment file and add your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
# OpenAI API Key (required for LLM extraction)
OPENAI_API_KEY=your-api-key-here

# Neo4j AuraDB (optional, for cloud persistence)
NEO4J_URI=neo4j+s://xxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password-here
```

### 3. Run the Application

```bash
cd /home/ifrodostein/work/bentonx/concrete-kg
streamlit run app/main.py
```

The app will open in your browser at `http://localhost:8501`.

## 📖 Usage Guide

### Demo Mode

Click **"Load Demo Data"** in the sidebar to explore the application with sample Portland Cement, Fly Ash, and Silica Fume data.

### Upload Research Papers

1. Click **"Upload Research Paper (PDF)"** in the sidebar
2. Select a PDF about concrete materials
3. Click **"Extract Knowledge"**
4. Wait for LLM to process the document

### Explore the Knowledge Graph

- **Extraction Results Tab**: View extracted materials and components in tables
- **Knowledge Graph Tab**: Interactive visualization with multiple layouts
- **Graph RAG Q&A Tab**: Ask natural language questions like:
  - "What is the CaO percentage in Portland cement?"
  - "Which components affect compressive strength?"
  - "What does silica fume contain?"
- **Export Tab**: Download as JSON or Cypher scripts

## 🏗️ Architecture

```
app/
├── main.py             # Streamlit web interface
├── config.py           # Configuration management  
├── pdf_processor.py    # PDF text extraction
├── llm_extractor.py    # LangChain extraction pipeline
├── knowledge_graph.py  # NetworkX + Neo4j graph engine
├── graph_rag.py        # Graph RAG query engine
└── visualizer.py       # Plotly visualizations
```

### Knowledge Graph Schema

**Node Types:**
- `Material`: Concrete mixtures (Portland Cement, Fly Ash, etc.)
- `Component`: Chemical compounds (CaO, SiO2, Al2O3, etc.)
- `Property`: Physical properties (Compressive Strength, Durability, etc.)
- `Source`: Research paper references

**Relationship Types:**
- `CONTAINS`: Material contains component (with percentage)
- `AFFECTS`: Component affects property
- `REACTS_WITH`: Components that react together
- `CITED_IN`: Referenced in source document

## ⚙️ Configuration

### LLM Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required for extraction |
| `OPENAI_API_BASE` | Custom API base URL | Optional (for local LLMs) |
| `LLM_MODEL` | Model to use | `gpt-4o-mini` |

### Neo4j Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `NEO4J_URI` | AuraDB connection URI | Optional |
| `NEO4J_USERNAME` | Database username | `neo4j` |
| `NEO4J_PASSWORD` | Database password | Optional |

## 🔧 Development

### Project Structure

```
concrete-kg/
├── app/                 # Application source code
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (not in git)
├── .env.example         # Environment template
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

### Running Tests

```bash
pyenv activate concrete-kg
cd /home/ifrodostein/work/bentonx/concrete-kg

# Test module imports
python -c "from app.pdf_processor import PDFProcessor; print('OK')"
python -c "from app.knowledge_graph import KnowledgeGraph; print('OK')"
python -c "from app.visualizer import Visualizer; print('OK')"
```

## 📚 Dependencies

- **langchain** - LLM orchestration framework
- **langchain-openai** - OpenAI integration
- **langchain-neo4j** - Neo4j graph database integration
- **networkx** - In-memory graph operations
- **neo4j** - Neo4j Python driver
- **pymupdf** - PDF text extraction
- **plotly** - Interactive visualizations
- **streamlit** - Web interface
- **pydantic** - Data validation

## 📄 License

This project uses open-source libraries only. See individual package licenses for details.
