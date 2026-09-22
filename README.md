# 🧠 Advance RAG with HyDE (Hypothetical Document Embeddings)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge&logo=chainlink&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Local%20VectorDB-0487D9?style=for-the-badge)

<p align="center">
  <b>Advanced Retrieval-Augmented Generation (RAG) System</b> implementing HyDE, Query Rewriting, and custom document loaders for high-precision domain-specific context retrieval.
</p>

</div>

---

## 📌 Executive Summary

**Advance RAG HyDE** is a specialized codebase engineered to bridge the semantic gap between user search intent and technical domain documents (e.g., insurance policy terms and claim procedures). By generating hypothetical documents via LLMs before vector lookup, this system significantly improves semantic matching accuracy over traditional naive RAG setups.

---

## ⚡ Key Features

| Feature | Description | Core Components |
| :--- | :--- | :--- |
| **🔮 HyDE Generator** | Generates hypothetical documents from user queries to improve dense vector retrieval. | `utils/hyde_generator.py` |
| **✍️ Query Rewriter** | Optimizes and expands raw queries into structured search statements. | `utils/query_rewriter.py` |
| **📂 Domain Document Loader** | Custom parsers for raw text corpora and structured knowledge repositories. | `utils/loader.py` |
| **🔍 Intelligent Retriever** | Handles multi-stage query execution and vector similarity scoring. | `utils/retriever.py` |
| **⚙️ Config-Driven Design** | Fully customizable parameters via YAML configuration and environment controls. | `config.yaml`, `.env` |

---

## 🏗️ Project Architecture

```
                               ┌───────────────────────────┐
                               │       User Query          │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │     Query Rewriter        │
                               │  (utils/query_rewriter.py)│
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │      HyDE Generator       │
                               │  (utils/hyde_generator.py)│
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │    Custom Retriever       │
                               │    (utils/retriever.py)   │
                               └─────────────┬─────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
          ┌─────────────────────────┐                 ┌─────────────────────────┐
          │     Vector Search       │                 │     Raw Documents       │
          │  (FAISS / Local Store)  │                 │ (insurance_docs/*.txt)  │
          └────────────┬────────────┘                 └────────────┬────────────┘
                       │                                           │
                       └─────────────────────┬─────────────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │     LLM Answer Gen        │
                               │        (app.py)           │
                               └───────────────────────────┘
```

---

## 📂 Project Structure

```
advance-rag-HyDE/
├── 📁 data/
│   └── 📁 raw/
│       └── 📁 insurance_docs/
│           ├── 📄 claim_procedure.txt     # Sample insurance claim guide
│           └── 📄 policy_terms.txt        # Sample policy terms and conditions
│
├── 📁 utils/                              # Core Processing Modules
│   ├── 📄 __init__.py
│   ├── 📄 loader.py                       # Document loader pipeline
│   ├── 📄 hyde_generator.py               # Hypothetical document embedding logic
│   ├── 📄 query_rewriter.py               # Query optimization module
│   ├── 📄 retriever.py                    # Search and retrieval orchestration
│   └── 📄 utils.py                        # Helper functions & shared utilities
│
├── 📄 app.py                              # Main application entry point
├── 📄 config.yaml                         # Central configuration parameters
├── 📄 template.py                         # Scaffold generator script
├── 📄 requirements.txt                    # Project dependencies
└── 📄 .env                                # Environment variables (API keys)
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have **Python 3.10+** installed.

### 2. Clone / Setup Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create or populate your `.env` file in the root directory with your preferred provider API keys:
```env
# Choose your LLM provider key(s):
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

### 5. Tune System Configuration (`config.yaml`)
Review or modify runtime parameters inside `config.yaml`:
```yaml
embedding_model: "text-embedding-3-small"
llm_model: "gpt-4o-mini"
chunk_size: 1000
chunk_overlap: 200
top_k: 3
```

---

## 💻 Running the Application

Execute the main pipeline or script:
```bash
python app.py
```

---

## 🤝 Contributing

1. Create a Feature Branch (`git checkout -b feature/HyDE-Enhancement`)
2. Commit your Changes (`git commit -m 'Add custom reranker support'`)
3. Push to the Branch (`git push origin feature/HyDE-Enhancement`)
4. Open a Pull Request