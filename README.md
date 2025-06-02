# 🔍 Advanced RAG Techniques Implementation Hub

A comprehensive repository showcasing state-of-the-art Retrieval-Augmented Generation (RAG) techniques with practical implementations, evaluation frameworks, and runnable scripts for production use.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-orange)](https://langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-blueviolet)](https://openai.com/)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [RAG Techniques Implemented](#rag-techniques-implemented)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Evaluation Framework](#evaluation-framework)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This repository provides a comprehensive collection of advanced RAG (Retrieval-Augmented Generation) techniques with both Jupyter notebook implementations for experimentation and production-ready Python scripts. The project is designed for researchers, developers, and practitioners working with large language models and information retrieval systems.

### Key Highlights

- **30+ RAG Techniques**: From basic implementations to cutting-edge approaches
- **Production Ready**: Runnable scripts optimized for real-world deployment
- **Comprehensive Evaluation**: Built-in metrics and evaluation frameworks
- **Multi-Modal Support**: Text, PDF, CSV, and image processing capabilities
- **Framework Agnostic**: Works with LangChain, LlamaIndex, and custom implementations

## ✨ Features

- 🚀 **Multiple RAG Implementations**: Simple RAG to advanced techniques like Graph RAG and Self-RAG
- 📊 **Built-in Evaluation**: Comprehensive evaluation metrics using RAGAS, DeepEval, and custom frameworks
- 🔧 **Production Scripts**: Ready-to-use Python scripts for all implemented techniques
- 📚 **Rich Documentation**: Detailed Jupyter notebooks with explanations and examples
- 🌐 **Multi-Provider Support**: OpenAI, Anthropic, Groq, Amazon Bedrock integration
- 📈 **Performance Optimization**: Advanced chunking, reranking, and retrieval strategies

## 🧠 RAG Techniques Implemented

### Core RAG Techniques
- **Simple RAG** - Basic retrieval-augmented generation
- **Semantic Chunking** - Content-aware text segmentation
- **Contextual Compression** - Dynamic context reduction
- **Fusion Retrieval** - Multi-query retrieval strategies

### Advanced Techniques
- **Self-RAG** - Self-reflective retrieval with feedback loops
- **Graph RAG** - Knowledge graph-based retrieval
- **RAPTOR** - Recursive abstractive processing
- **CRAG** - Corrective retrieval-augmented generation
- **Adaptive Retrieval** - Dynamic retrieval strategies

### Multi-Modal RAG
- **Multi-Modal RAG with ColPali** - Visual document understanding
- **Multi-Modal RAG with Captioning** - Image-to-text retrieval

### Optimization Techniques
- **Reranking** - Advanced result reordering
- **Query Transformations** - Query expansion and refinement
- **Hierarchical Indices** - Multi-level document organization
- **Document Augmentation** - Content enhancement strategies

### Specialized Approaches
- **HyDE** - Hypothetical Document Embeddings
- **HyPE** - Hypothetical Prompt Embeddings
- **Contextual Chunk Headers** - Enhanced chunk metadata
- **Explainable Retrieval** - Interpretable retrieval decisions

## 📁 Project Structure

```
RAG_System/
├── all_rag_techniques/              # Jupyter notebooks with detailed implementations
│   ├── simple_rag.ipynb
│   ├── graph_rag.ipynb
│   ├── self_rag.ipynb
│   └── ...
├── all_rag_techniques_runnable_scripts/  # Production-ready Python scripts
│   ├── simple_rag.py
│   ├── graph_rag.py
│   ├── self_rag.py
│   └── ...
├── evaluation/                      # Evaluation frameworks and metrics
│   ├── evalute_rag.py
│   ├── evaluation_deep_eval.ipynb
│   └── evaluation_grouse.ipynb
├── data/                           # Sample datasets and documents
│   ├── nike_2023_annual_report.txt
│   ├── Understanding_Climate_Change.pdf
│   ├── customers-100.csv
│   └── q_a.json
├── tests/                          # Unit tests and integration tests
├── images/                         # Documentation images and diagrams
├── helper_functions.py             # Utility functions and shared components
└── README.md
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/BOB0320/RAG_System.git
cd RAG_System
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Create a .env file with your API keys
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
echo "ANTHROPIC_API_KEY=your_anthropic_api_key_here" >> .env
```

### Required Dependencies

```bash
pip install langchain langchain-openai langchain-community
pip install faiss-cpu  # or faiss-gpu for GPU support
pip install openai anthropic groq
pip install pandas numpy matplotlib seaborn
pip install jupyter notebook
pip install rank-bm25 pypdf sentence-transformers
pip install ragas deepeval  # for evaluation
```

## 🏃‍♂️ Quick Start

### 1. Basic RAG Implementation

```python
from all_rag_techniques_runnable_scripts.simple_rag import SimpleRAG

# Initialize RAG system
rag = SimpleRAG()

# Process documents
vectorstore = rag.process_documents("data/Understanding_Climate_Change.pdf")

# Ask questions
response = rag.query("What are the main causes of climate change?")
print(response)
```

### 2. Advanced Graph RAG

```python
from all_rag_techniques_runnable_scripts.graph_rag import GraphRAG

# Initialize Graph RAG
graph_rag = GraphRAG()

# Build knowledge graph and query
result = graph_rag.process_and_query(
    document_path="data/nike_2023_annual_report.txt",
    query="What were Nike's key financial highlights in 2023?"
)
```

### 3. Self-Reflective RAG

```python
from all_rag_techniques_runnable_scripts.self_rag import SelfRAG

# Initialize Self-RAG with feedback loops
self_rag = SelfRAG()

# Query with self-correction
answer = self_rag.query_with_reflection(
    "Explain the impact of climate change on biodiversity",
    max_iterations=3
)
```

## 📊 Usage Examples

### Jupyter Notebook Exploration

```bash
# Start Jupyter server
jupyter notebook

# Navigate to all_rag_techniques/ directory
# Open any notebook to explore implementations
```

### Running Production Scripts

```python
# Example: Running semantic chunking
from all_rag_techniques_runnable_scripts.semantic_chunking import SemanticChunking

chunker = SemanticChunking(chunk_size=512)
chunks = chunker.process_document("data/nike_2023_annual_report.txt")
```

### Evaluation Pipeline

```python
from evaluation.evalute_rag import evaluate_rag_system

# Evaluate multiple RAG techniques
results = evaluate_rag_system(
    techniques=['simple_rag', 'graph_rag', 'self_rag'],
    test_dataset="data/q_a.json",
    metrics=['faithfulness', 'relevancy', 'context_precision']
)
```

## 🔬 Evaluation Framework

The repository includes comprehensive evaluation tools:

- **RAGAS Integration**: Automated evaluation using RAGAS metrics
- **DeepEval Support**: Advanced evaluation with DeepEval framework
- **Custom Metrics**: Domain-specific evaluation criteria
- **Comparative Analysis**: Side-by-side technique comparison

### Running Evaluations

```python
# Run complete evaluation suite
python evaluation/evalute_rag.py --config evaluation_config.json

# Generate evaluation reports
jupyter notebook evaluation/evaluation_deep_eval.ipynb
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Ensure all tests pass: `python -m pytest tests/`
5. Submit a pull request

### Adding New RAG Techniques

1. Create a Jupyter notebook in `all_rag_techniques/`
2. Implement the corresponding Python script in `all_rag_techniques_runnable_scripts/`
3. Add appropriate tests in `tests/`
4. Update documentation

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the excellent framework
- [LlamaIndex](https://llamaindex.ai/) for advanced indexing capabilities
- [OpenAI](https://openai.com/) for powerful language models
- The open-source community for inspiration and contributions

## 📞 Support

- 📧 Email: diamuji1@gmail.com
- 💬 GitHub Issues: [Report a bug or request a feature](https://github.com/yourusername/RAG_System/issues)
- 📖 Documentation: [Full documentation](https://yourusername.github.io/RAG_System/)

---

⭐ **Star this repository if you find it helpful!** ⭐

*Built with ❤️ for the AI and ML community* 