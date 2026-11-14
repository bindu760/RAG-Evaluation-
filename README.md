# RAG Evaluation & Benchmarking System

An AI-powered application designed to evaluate, analyze, and benchmark Retrieval-Augmented Generation (RAG) systems using different evaluation metrics and configurations.

## Overview

The RAG Evaluation & Benchmarking System helps measure the performance, accuracy, reliability, and efficiency of RAG applications.

It allows users to upload PDF documents, build a RAG pipeline, ask questions, generate AI-powered responses, and evaluate the quality of generated answers using automated evaluation techniques.

The system also compares different retrieval configurations to identify the best-performing RAG setup.

## Objectives

- Evaluate the accuracy and reliability of RAG systems.
- Measure the quality of retrieved document chunks.
- Detect hallucinations in generated answers.
- Analyze retrieval and generation performance.
- Compare different chunk sizes and retrieval configurations.
- Generate evaluation reports and visualizations.

## Key Features

- **PDF Document Upload:** Upload documents to create a knowledge base.
- **Document Processing:** Extract text and split documents into chunks.
- **Embedding Generation:** Convert text chunks into vector embeddings.
- **Vector Search:** Store and retrieve relevant information using FAISS.
- **RAG Pipeline:** Generate context-aware answers using an LLM.
- **Automated Evaluation:** Evaluate RAG responses using Ragas.
- **Retrieval Evaluation:** Measure Precision@K, Recall@K, and other retrieval metrics.
- **Hallucination Analysis:** Check whether generated answers are grounded in retrieved context.
- **Performance Benchmarking:** Compare different RAG configurations.
- **Latency Measurement:** Measure retrieval time and answer generation time.
- **Evaluation Dashboard:** Visualize evaluation scores and performance metrics.
- **Report Generation:** Export evaluation results for further analysis.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| LangChain | RAG pipeline development |
| FAISS | Vector database |
| Hugging Face | Embedding models |
| Groq API | LLM inference |
| Ragas | RAG evaluation |
| Pandas | Data processing |
| Matplotlib | Data visualization |
| PyPDF | PDF document extraction |

## RAG Evaluation Metrics

The system will evaluate RAG performance using the following metrics:

### Retrieval Metrics
- Precision@K
- Recall@K
- Hit Rate@K
- Mean Reciprocal Rank (MRR)
- Context Precision
- Context Recall

### Generation Metrics
- Faithfulness
- Answer Relevancy
- Answer Correctness
- Context Relevancy
- Hallucination Analysis

### Performance Metrics
- Retrieval Latency
- Generation Latency
- Total Response Time
- Token Usage

## Project Architecture

```text
PDF Documents
     |
     v
Document Processing
     |
     v
Text Chunking
     |
     v
Embedding Generation
     |
     v
FAISS Vector Database
     |
     v
Relevant Context Retrieval
     |
     v
LLM Answer Generation
     |
     v
RAG Evaluation (Ragas)
     |
     v
Performance Benchmarking
     |
     v
Streamlit Evaluation Dashboard
```

## Project Structure

```text
rag-evaluation-system/
│
├── data/
│   ├── documents/
│   └── evaluation_dataset.csv
│
├── src/
│   ├── document_loader.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── rag_pipeline.py
│   └── evaluator.py
│
├── evaluation/
│   └── results/
│
├── app.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd rag-evaluation-system
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Environment

Windows:

```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

### 6. Run the Application

```bash
streamlit run app.py
```

## Expected Outcomes

- A functional RAG evaluation application.
- Automated evaluation of generated answers.
- Quantitative measurement of retrieval quality.
- Identification of hallucinations and retrieval failures.
- Performance comparison across different RAG configurations.
- Interactive evaluation dashboard and reports.

## Development Roadmap

- [ ] Project initialization and environment setup.
- [ ] PDF document loading and processing.
- [ ] Text chunking and embedding generation.
- [ ] FAISS vector database integration.
- [ ] RAG pipeline implementation.
- [ ] Evaluation dataset preparation.
- [ ] Ragas evaluation integration.
- [ ] Retrieval metrics implementation.
- [ ] Latency and performance benchmarking.
- [ ] Streamlit dashboard development.
- [ ] Evaluation report generation.
- [ ] Testing and deployment.

## License

This project is developed for educational and portfolio purposes.
