<div align="center">

# 📚 StudyAI: AI-Powered Study Assistant

### *An Intelligent Retrieval-Augmented Generation (RAG) System for Exam Preparation*

> **Powered by a complete RAG Pipeline that transforms PDFs into structured exam-ready notes, enables context-aware conversations through semantic retrieval, and reinforces learning with automatically generated MCQ quizzes.**

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/RAG%20Pipeline-Retrieval%20Augmented%20Generation-00C853?style=for-the-badge">
  <img src="https://img.shields.io/badge/FAISS-Vector%20Database-blueviolet?style=for-the-badge">
  <img src="https://img.shields.io/badge/SentenceTransformers-Semantic%20Embeddings-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/LLM-AI%20Powered-red?style=for-the-badge">
  <img src="https://img.shields.io/badge/Streamlit-Web%20Application-FF4B4B?style=for-the-badge">
</p>

### StudyAI is more than a notes generator—it is a complete AI learning companion that combines document understanding, semantic retrieval, conversational AI, and automated assessment into a single interactive platform.

</div>

---

# Project Preview

<p align="center">
<img src="assets/homepage.png" width="95%">
</p>

> Replace with the main screenshot of your application.

---

# Overview

Studying from lengthy PDFs often involves manually extracting important concepts, creating notes, revising key topics, and testing understanding through practice questions. This repetitive workflow consumes significant time and frequently reduces learning efficiency.

**StudyAI** automates this entire process through a **Retrieval-Augmented Generation (RAG) Pipeline**, allowing users to upload academic documents and instantly transform them into structured revision material while interacting with the content through natural language.

Unlike conventional summarization tools, StudyAI first understands the document, builds a semantic knowledge base, retrieves relevant information based on meaning rather than keywords, and then generates context-aware responses grounded in the uploaded material.

The result is an intelligent study assistant capable of supporting every stage of exam preparation—from note generation and doubt solving to self-assessment and revision.

---

# **Retrieval-Augmented Generation (RAG) Pipeline**

The foundation of StudyAI is a **Retrieval-Augmented Generation (RAG) Pipeline**, designed to overcome the limitations of standalone language models by combining semantic retrieval with generative AI.

Instead of relying only on pretrained knowledge, the system retrieves the most relevant information from the uploaded document before generating a response.

The complete RAG workflow consists of:

- PDF text extraction
- Intelligent document chunking
- Semantic embedding generation
- FAISS vector indexing
- Similarity-based retrieval
- Context injection into the language model
- AI-generated context-aware responses

By grounding every answer in the uploaded document, the system produces responses that are significantly more reliable, relevant, and personalized for the user's study material.


#**RAG Pipeline Architecture**

```text
              User Question
                    │
                    ▼
      Convert Query to Embedding
                    │
                    ▼
      Semantic Similarity Search
           (FAISS Vector Store)
                    │
                    ▼
     Retrieve Most Relevant Chunks
                    │
                    ▼
      Context + User Prompt + LLM
                    │
                    ▼
      Context-Aware AI Response
```
      
---

# Why StudyAI?

StudyAI is designed as an end-to-end AI learning platform rather than a simple summarization tool.

It combines multiple AI capabilities into a unified workflow:

- **RAG-powered contextual question answering**
- **AI-generated exam-ready notes**
- **Semantic search using dense vector embeddings**
- **Interactive chatbot grounded in uploaded PDFs**
- **Automatic MCQ quiz generation**
- **Performance evaluation and scoring**
- **Exportable notes for offline revision**

This integrated approach encourages active learning while dramatically reducing the effort required to prepare for examinations.

---

# Key Features

## AI-Generated Exam Notes

Transform lengthy academic documents into structured revision material optimized for examination preparation.

Instead of producing generic summaries, StudyAI organizes information into concise, readable, and hierarchical notes that emphasize the concepts most likely to matter during revision.

---

## **RAG-Powered Conversational AI**

At the heart of StudyAI lies a **Retrieval-Augmented Generation (RAG) Pipeline**.

Whenever a user asks a question:

1. The query is converted into an embedding.
2. FAISS retrieves the most semantically relevant document chunks.
3. Retrieved context is supplied to the language model.
4. The model generates a response grounded in the uploaded study material.

This architecture significantly reduces hallucinations while producing highly contextual and document-specific answers.

---

## Semantic Search with FAISS

Every uploaded PDF is converted into dense vector representations using Sentence Transformers.

These embeddings are stored inside a FAISS vector database, enabling similarity search based on meaning rather than exact keyword matches.

As a result, users can ask natural language questions and still retrieve the most relevant portions of their study material.

---

## Interactive AI Tutor

StudyAI allows users to converse naturally with their documents.

Whether requesting explanations, clarifications, definitions, or deeper insights, the chatbot retrieves relevant information before generating its response, creating an experience similar to interacting with a knowledgeable tutor.

---

## Automated MCQ Generation

To reinforce learning, StudyAI automatically generates multiple-choice questions based on the generated notes.

These quizzes encourage active recall, identify weak areas, and provide immediate evaluation to improve retention.

---

## PDF Export

Generated notes can be exported as PDF documents, allowing students to retain organized revision material for offline study and future reference.

# System Workflow

```text
                     User Uploads PDF
                             │
                             ▼
                  PDF Text Extraction
                             │
                             ▼
                  Text Cleaning & Processing
                             │
                             ▼
                    Intelligent Chunking
                             │
            ┌────────────────┴────────────────┐
            │                                 │
            ▼                                 ▼
   Generate Exam Notes          Generate Sentence Embeddings
            │                                 │
            │                                 ▼
            │                    Build FAISS Vector Index
            │                                 │
            │                                 │
            ▼                                 ▼
     Export Notes as PDF           User Asks a Question
                                              │
                                              ▼
                                  Convert Query to Embedding
                                              │
                                              ▼
                                Retrieve Similar Chunks (FAISS)
                                              │
                                              ▼
                              Pass Context + Query to LLM
                                              │
                                              ▼
                               Generate Context-Aware Answer
                                              │
                                              ▼
                             AI Chat + MCQ Quiz Generation
```
# Installation

## Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/StudyAI.git
```

Navigate to the project directory:

```bash
cd StudyAI
```

---

## Install Dependencies

Install all required packages using:

```bash
pip install -r requirements.txt
```

---

## Run the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

Once started, the application will automatically open in your browser.

Alternatively, you can use the deployed version directly:

**Live Demo:**  
`https://your-streamlit-link-here`

---

# Usage

### 1. Upload a PDF

Begin by uploading your study material or lecture notes in PDF format.

---

### 2. AI Processing

The application automatically:

- Extracts text from the document
- Cleans and preprocesses the content
- Splits it into semantic chunks
- Generates dense vector embeddings
- Builds a FAISS vector database
- Creates structured exam-ready notes

---

### 3. Read Exam Notes

Review concise, organized notes that highlight the most important concepts from the uploaded material.

---

### 4. Chat with Your Document

Ask questions in natural language.

The RAG pipeline retrieves the most relevant sections from your uploaded PDF before generating an answer, ensuring responses remain grounded in your study material.

---

### 5. Test Yourself

Generate AI-powered MCQ quizzes based on the processed notes and evaluate your understanding with instant feedback.

---

### 6. Export Notes

Download generated notes as a PDF for offline revision and future reference.

---

# Project Structure

```text
StudyAI/
│
├── app.py
├── requirements.txt
├── README.md
├── assets/
│   ├── homepage.png
│   ├── notes.png
│   ├── chat.png
│   ├── quiz.png
│   └── architecture.png
│
├── uploads/
├── vector_store/
└── temp/
```

---

# Technology Stack

## Programming Language

- Python

## Frontend

- Streamlit

## Large Language Model

- Groq API
- Llama 3

## Retrieval-Augmented Generation

- FAISS Vector Database
- Sentence Transformers
- Semantic Similarity Search

## Natural Language Processing

- LangChain Text Splitters
- Context Retrieval
- Embedding Generation

## PDF Processing

- PyMuPDF (fitz)

## Data Processing

- NumPy

## Visualization

- Matplotlib

## Document Generation

- FPDF

---

# Core AI Technologies

StudyAI integrates multiple modern AI techniques into a single application:

- **Retrieval-Augmented Generation (RAG)**
- **Semantic Search**
- **Vector Embeddings**
- **Dense Vector Indexing with FAISS**
- **Context-Aware Conversational AI**
- **Large Language Models**
- **Automatic Note Generation**
- **AI-Powered Quiz Generation**
- **Document Grounding**
- **Interactive Learning**

Together, these components enable StudyAI to move beyond traditional summarization and function as an intelligent learning companion.

---

# Why Retrieval-Augmented Generation?

Traditional language models answer questions based solely on their pretrained knowledge, which may lead to hallucinations or outdated information.

StudyAI addresses this challenge through a **Retrieval-Augmented Generation (RAG) Pipeline**.

Before generating a response, the system:

1. Searches the uploaded document semantically.
2. Retrieves the most relevant chunks.
3. Injects that context into the language model.
4. Produces an answer grounded in the user's own study material.

This significantly improves factual consistency and ensures responses remain personalized to the uploaded document.

---

# Advantages of StudyAI

Compared to conventional note-taking or summarization tools, StudyAI offers several advantages:

| Traditional Tools | StudyAI |
|-------------------|----------|
| Static summaries | Dynamic AI-generated notes |
| Keyword search | Semantic search |
| No contextual understanding | RAG-powered contextual retrieval |
| No interaction | Conversational AI assistant |
| Manual quiz preparation | Automatic MCQ generation |
| Passive learning | Active learning through chat and quizzes |
| Fixed documents | Personalized document understanding |

---

# Future Enhancements

StudyAI has been designed with extensibility in mind.

Potential future improvements include:

- Multi-PDF knowledge bases
- OCR support for scanned documents
- Image and diagram understanding
- Citation-aware answers
- Flashcard generation
- Adaptive quizzes based on user performance
- Voice-based interaction
- Study progress tracking
- Personalized revision schedules
- Multi-language support

---

# Learning Outcomes

This project demonstrates practical implementation of:

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- FAISS Vector Databases
- Sentence Embeddings
- Document Understanding
- Large Language Model Integration
- Conversational AI
- Prompt Engineering
- Context Retrieval
- Exam-Oriented AI Systems
- PDF Processing
- Interactive Web Applications

It showcases how modern AI techniques can be combined to create an end-to-end intelligent learning platform rather than a standalone summarization tool.

---

# Who Is It For?

StudyAI is designed for:

- Students preparing for examinations
- University learners
- Competitive exam aspirants
- Researchers reviewing academic papers
- Professionals studying technical documentation
- Anyone who wants to interact with documents instead of simply reading them

---

# License

This project is released under the MIT License.

You are free to use, modify, and distribute it in accordance with the license terms.

---

# Acknowledgements

This project builds upon advancements in:

- Retrieval-Augmented Generation (RAG)
- Sentence Transformers
- FAISS Vector Search
- Large Language Models
- Streamlit
- Modern NLP research

---

# Disclaimer

StudyAI is intended for educational and research purposes.

While it strives to provide accurate and context-aware responses, generated notes and answers should be verified against original source material, particularly in academic or professional settings.

---

<div align="center">

# StudyAI

### **An AI-Powered Study Assistant built around a complete Retrieval-Augmented Generation (RAG) Pipeline**

*Upload PDFs • Generate Exam-Ready Notes • Chat with Your Documents • Practice with AI-Generated MCQs*

---

If you found this project useful, consider giving it a ⭐ to support its development.

</div>
