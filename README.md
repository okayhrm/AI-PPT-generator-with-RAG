# AI-Powered PowerPoint Generator with RAG Support

## Overview

This project is an AI-assisted presentation generation system that allows users to create and edit PowerPoint slides through natural language interaction. The system integrates Retrieval-Augmented Generation (RAG), enabling users to upload reference documents and generate slide content based on that material. Users can also select the level of detail (light, detailed, or extensive) and choose from multiple slide templates.

The project includes a conversational interface for iterative refinement, allowing users to revise slide content, shorten or expand explanations, change tone, or adjust formatting through simple chat-based instructions.

---

## Key Features

* **Natural Language Slide Generation:** Create presentations by describing the topic in plain text.
* **Retrieval-Augmented Generation (RAG):** Upload PDFs and generate slide content grounded in user-provided material.
* **Template Selection:** Choose from multiple built-in PowerPoint templates for consistent design.
* **Adjustable Content Depth:** Select light, detailed, or extensive slide coverage.
* **Conversational Editing:** Modify slides through continuous chat interaction without regenerating from scratch.
* **PowerPoint Export:** Automatically compiles and downloads `.pptx` files using `python-pptx`.

---

## System Architecture

### NLP Components

| Function                       | Technique Used                                 |
| ------------------------------ | ---------------------------------------------- |
| Intent Detection               | Prompt parsing and rule-based interpretation   |
| Topic Extraction               | Keyword extraction and semantic similarity     |
| Document Retrieval             | Vector embeddings + Pinecone similarity search |
| Summarization & Slide Drafting | Abstractive generation using Gemini model      |
| Editing & Transformation       | Iterative rewriting and style adjustment       |
| Context Preservation           | Session memory within Streamlit interface      |

### Workflow Diagram

```
User Prompt → Determine Mode (RAG / Generation) → Retrieve Relevant Context (if RAG enabled)
→ Generate Structured Slide Content → User Reviews and Edits → Export to PowerPoint
```

---

## Tech Stack

| Component       | Implementation                            |
| --------------- | ----------------------------------------- |
| Frontend UI     | Streamlit                                 |
| LLM             | Google Gemini (text and embedding models) |
| Vector Database | Pinecone                                  |
| Slide Rendering | python-pptx                               |
| Orchestration   | LangGraph State Machine                   |
| Storage         | Local filesystem templates                |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/okayhrm/AI-PPT-generator-with-RAG.git
cd AI-PPT-generator-with-RAG
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=ppt-rag
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
EMBED_PROVIDER=google
EMBED_DIM=768
```

Ensure your Gemini key has access to:

* Generative Language API
* Embedding model `models/embedding-001`

---

## Running the Application

```bash
streamlit run app.py
```

The application will launch in your browser at:

```
http://localhost:8501
```

---

## Usage Instructions

1. Enter the topic or project description.
2. Choose the slide detail level (light / detailed / extensive).
3. Upload reference PDF documents if RAG grounding is desired.
4. Generate slides and review the content.
5. Modify slide content conversationally through the editing interface.
6. Export to PowerPoint.

---

## Directory Structure

```
.
├── app.py                  # Streamlit user interface
├── ppt_gen.py              # LangGraph workflow and slide generation logic
├── embedder.py             # Embedding model selector (Gemini / fallback)
├── ingest_pinecone.py      # PDF/Document ingestion and vector storage
├── templates/              # PowerPoint base templates and thumbnails
├── requirements.txt
└── README.md
