# 🧠 Exam Guide — AI-Powered Study Assistant

---

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![Groq](https://img.shields.io/badge/Groq-Llama%203.1-orange)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-purple)

</p>

---

## 📌 Overview

**Exam Guide** is an AI-powered document intelligence system designed to transform unstructured PDF content into an interactive learning environment.

The system enables:
- Structured exam note generation  
- Context-aware question answering  
- Automated MCQ generation and evaluation  
- Exportable study material  

It is built on a **retrieval-augmented semantic search pipeline combined with large language model reasoning**, ensuring responses are grounded in the uploaded document rather than generated from memory.
<p align="center">
  <img src="assets/homepage1.png" width="85%">
</p>
---

## ⚙️ System Requirements

### 🔑 Groq API Key (Mandatory)

A valid Groq API key is required to enable LLM-based features.

- Entered via the Streamlit sidebar  
- Required for:
  - Notes generation  
  - Chat system  
  - MCQ generation  

Without it, the system operates only as a document viewer.

---

## 📄 Document Processing Strategy

When a PDF is uploaded, the system follows a controlled processing pipeline:

### Step 1: Text Extraction
- Extracted using PyMuPDF (`fitz`)
- Raw text is cleaned (whitespace normalization, line merging)

---

### Step 2: Controlled Chunking Strategy

The document is split using:

- `chunk_size = 2000 tokens (approx)`
- `chunk_overlap = 200`
- Maximum processing limit:
  MAX_CHUNKS_TO_PROCESS = 5
  
### Why this matters:
- Prevents API overload
- Ensures fast inference
- Reduces embedding + LLM cost
- Keeps retrieval focused on high-signal sections

Only the **first 5 chunks** are processed for downstream tasks.

---

### Step 3: Embedding & Indexing

- Model: `all-MiniLM-L6-v2`
- Embeddings are normalized (L2 normalization)
- Stored in FAISS IndexFlatIP (cosine similarity equivalent)

This enables:
- Fast semantic search
- Efficient similarity retrieval
- Lightweight vector storage

---

## 🧠 System Architecture 

Unlike a linear flowchart, the system is designed as a **multi-stage retrieval + generation pipeline with branching execution paths**.


## Architecture Diagram
```text
                     ┌─────────────────────┐
                     │     PDF Upload      │
                     └─────────┬───────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │  PyMuPDF Text Extraction │
                 └─────────┬────────────────┘
                               │
                               ▼
              ┌──────────────────────────────┐
              │ Text Cleaning & Normalization│
              └─────────┬────────────────────┘
                               │
                               ▼
     ┌────────────────────────────────────────────┐
     │ Recursive Character Text Chunking          │
     │ (Chunk Size: 2000 | Overlap: 200)          │
     └─────────┬──────────────────────────────────┘
                               │
                               ▼
    ┌─────────────────────────────────────────────┐
    │ Sentence Transformer Embeddings             │
    │ (all-MiniLM-L6-v2)                          │
    └─────────┬───────────────────────────────────┘
                               │
                               ▼
    ┌─────────────────────────────────────────────┐
    │ FAISS Vector Index (Cosine Similarity)      │
    └─────────┬───────────────────────────────────┘
                               │
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
 ┌─────────────────┐ ┌────────────────┐ ┌──────────────────┐
 │ Notes Generator  │ │ RAG Chat System │ │ MCQ Generator  │
 └────────┬────────┘ └────────┬───────┘ └────────┬─────────┘
          │                  │                   │
          ▼                  ▼                   ▼
 ┌────────────────────────────────────────────────────┐
 │     Groq Llama 3.1 8B Instant (LLM Layer)          │
 └────────────────────────────────────────────────────┘
```

---

## 🧠 Prompt Engineering (System Design Perspective)

Instead of treating prompts as isolated text blocks, this system uses **role-based prompt control strategies**.

---

### 1. Instruction-Constrained Generation (Notes System)

The model is guided to:
- Transform raw text into structured academic output
- Remove noise and redundancy
- Preserve only exam-relevant information

This ensures consistent **study-grade formatting** without manual rules.
<p align="center">
  <img src="assets/notes.png" width="60%">
</p>
---

### 2. Context-Grounded QA (Chat System)

The chatbot is strictly constrained to:

- Use ONLY retrieved FAISS context
- Avoid external knowledge injection
- Return fallback response when context is insufficient

This reduces hallucination and ensures **document-bound reasoning**.
<p align="center">
  <img src="assets/chat.png" width="60%">
</p>
---

### 3. Structured Output Enforcement (MCQ System)

MCQ generation is designed as a **schema-constrained task**:

- Output must follow strict JSON structure
- Deterministic formatting required for parsing
- Includes validation layer for malformed outputs
<p align="center">
  <img src="assets/quiz.png" width="60%">
</p>
---

### 🧩 SafeJSON Handling Layer

The system includes a **Safe JSON recovery mechanism**:

- Removes markdown code blocks from LLM output
- Extracts JSON using regex fallback
- Attempts parsing with fallback extraction logic

This ensures:
- MCQ system does not break due to malformed LLM responses  
- Robust handling of unpredictable LLM formatting  

---

## 🧠 Chunking Strategy (Critical Design Component)

The system uses controlled chunking:

- Chunk size: ~2000 characters  
- Overlap: 200 characters  
- Max chunks processed: **5 only**

### Design Reasoning:
- Prevents token overflow in LLM calls  
- Reduces embedding computation cost  
- Improves retrieval relevance  
- Forces focus on high-information segments  

This is a **performance + cost optimization decision**, not just preprocessing.

---

## 📂 Project Structure
ExamGuide/
│
├── notes_app.py 
├── notes.ipynb
├── requirements.txt 
├── README.md
│
├── assets
│ ├── homepage1.png 
│ ├── notes.png 
│ ├── chat.png 
│ ├── quiz.png 

## ⚙️ Tech Stack

The project is built using a lightweight but scalable AI stack optimized for document-based retrieval and generation.

- **Application Framework:** Streamlit  
- **LLM API:** Groq (Llama 3.1 Instant)  
- **Embedding Model:** SentenceTransformers (all-MiniLM-L6-v2)  
- **Vector Database:** FAISS (Facebook AI Similarity Search)  
- **PDF Processing:** PyMuPDF (fitz)  
- **Text Chunking:** LangChain Recursive Character Text Splitter  
- **Backend Language:** Python 3.10+  
- **Output Formatting:** FPDF (PDF generation)  
- **Data Reliability Layer:** Regex-based SafeJSON parsing  

---

## 🖥️ Streamlit Application Design

The entire system runs as a **single-page interactive Streamlit application**.

### UI Structure:
- Sidebar:
  - Groq API key input
  - PDF upload section
  - “Process Document” trigger
- Main Interface:
  - Tab 1: Notes
  - Tab 2: Chat with Document
  - Tab 3: MCQ Quiz

### State Management:
- Uses `st.session_state` for:
  - Uploaded chunks
  - Embeddings model
  - FAISS index
  - Chat history
  - Quiz answers

This ensures a **persistent session experience without database dependency**.

## ⚙️ How the System Works

The application follows a **Retrieval-Augmented Document Intelligence pipeline** built inside a Streamlit web interface.

---

### 📥 1. Document Upload (Streamlit UI)

- User uploads a PDF file through the Streamlit sidebar
- The file is temporarily stored and passed to the processing pipeline

---

### 📄 2. Text Extraction & Cleaning

- PDF text is extracted using **PyMuPDF (fitz)**
- Raw text is cleaned to remove:
  - Extra spaces  
  - Broken lines  
  - Irrelevant formatting noise  

This ensures consistent downstream processing.

---

### ✂️ 3. Chunking Strategy

The cleaned document is split into smaller overlapping segments:

- Chunk size: ~2000 characters  
- Overlap: ~200 characters  
- Processing limit: **only first 5 chunks are used**

This is a deliberate design choice to:
- Reduce API cost  
- Improve response speed  
- Focus on high-information content  

---

### 🧠 4. Embedding Generation

- Each chunk is converted into vector embeddings using:
  - `SentenceTransformers (all-MiniLM-L6-v2)`

These embeddings capture semantic meaning rather than keyword matching.

---

### 📦 5. Vector Storage (FAISS)

- All embeddings are stored in a FAISS index
- Enables fast similarity search using cosine similarity (via L2 normalization)

This allows the system to retrieve relevant content efficiently during queries.

---

## 💬 6. Core Functional Modules

Once the document is processed, the system unlocks three main features:

---

### 📝 A. Notes Generation Module

- Each chunk is sent to the Groq LLM
- The model converts raw text into structured exam notes
- Output focuses on:
  - Key concepts  
  - Bullet points  
  - Exam-oriented formatting  

Result: condensed study material from large PDFs.

---

### 💬 B. Chat with Document (RAG System)

This is the **core retrieval system**.

**Flow:**
1. User asks a question  
2. Question is converted into an embedding  
3. FAISS retrieves most relevant chunks  
4. Retrieved context + question is sent to Groq LLM  
5. LLM generates an answer strictly based on context  

If context is insufficient, the model returns a fallback-style response.

---

### 🧩 C. MCQ Quiz Generator

- Document content is sent to LLM in batches
- Model generates multiple-choice questions
- Output is expected in structured JSON format

A **SafeJSON layer** ensures:
- Removal of markdown artifacts  
- Recovery from malformed JSON  
- Stable parsing for quiz rendering  

---

## 📊 Output & Evaluation Layer

### Quiz System
- Displays MCQs in Streamlit UI
- Captures user answers using session state
- Calculates:
  - Score
  - Percentage accuracy
  - Performance feedback  

---

### 📄 PDF Export
- Notes are formatted using FPDF
- Generates downloadable revision document
- Ensures clean offline study material

---
## 🚀 Live Demo

Try the deployed application here:

👉https://notesmaker-mdhia3eskryyji43xhlgfb.streamlit.app/

---

## 🚧 Limitations

While the system is functional and optimized, it has some constraints:

- Only first **5 chunks are processed per document**
  - limits full-document reasoning for very large PDFs  
- No persistent storage (session-based only)
- FAISS index is in-memory (not saved across sessions)
- Requires active Groq API key for all LLM operations
- No multi-document comparison capability

---

## 🚀 Future Scope

This system can be extended into a production-grade AI learning platform.

### 1. Multi-Document RAG System
- Support multiple PDFs simultaneously
- Cross-document question answering

---

### 2. Persistent Vector Database
- Replace in-memory FAISS with:
  - Pinecone / ChromaDB / FAISS disk storage
- Enable long-term document memory

---

### 3. Advanced Retrieval Ranking
- Add reranking models for better chunk selection
- Improve context precision in QA system

---

### 4. Streaming Responses
- Real-time token streaming in chat interface
- Better UX for long answers

---

### 5. Adaptive MCQ Difficulty System
- Dynamic difficulty based on user performance
- Personalized learning paths

---

### 6. AI-Powered Study Assistant Expansion
- Voice-based Q&A system
- Smart revision scheduler
- Auto-generated revision summaries

---

### 7. Deployment Enhancements
- Streamlit Cloud / HuggingFace Spaces deployment
- Dockerized production build

---

## ⚠️ System Constraints Summary

- Designed for single-user session usage
- Optimized for speed over full-document reasoning
- Depends on external LLM API (Groq)
- No offline AI inference capability

---

## 📌 Final Summary

Exam Guide is a **Streamlit-based AI document intelligence system** that combines:

- Semantic search (FAISS)
- Lightweight embeddings
- LLM-based reasoning (Groq)
- Structured study output generation

It demonstrates a complete **retrieval-augmented generation pipeline for educational content processing**.

---
