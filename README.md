# Specialized-LLM-Bot-Using-Pre-Trained-Models
industry-specific Large Language Model (LLM) Bot using pre-trained models from platforms such as Hugging Face

# 🤖 Kreative Kudi AI Assistant

An AI-powered knowledge assistant developed for **Kreative Kudi** using **Retrieval-Augmented Generation (RAG)**.

The system allows users to ask questions about Kreative Kudi's services, courses, founder, location, pricing, contact information, and other company-related information. Instead of relying entirely on the language model's general knowledge, the assistant retrieves relevant information from a custom knowledge base created from the Kreative Kudi website.

---

## 📌 Project Overview

The **Kreative Kudi AI Assistant** is a Retrieval-Augmented Generation based chatbot designed to provide accurate and context-aware answers about Kreative Kudi.

The project follows a complete data-to-chatbot pipeline:

```text
Kreative Kudi Website
        ↓
Website Crawling
        ↓
Data Cleaning & Preprocessing
        ↓
Document Chunking
        ↓
Sentence Transformer
        ↓
Semantic Embeddings
        ↓
FAISS Vector Index
        ↓
Relevant Document Retrieval
        ↓
Gemini Generative Model
        ↓
AI Assistant Response
`````


<p align="center">
  <img src="https://github.com/Navjotkhatri/Specialized-LLM-Bot-Using-Pre-Trained-Models/blob/main/Screenshot%202026-08-22%20202336.png?raw=true" alt="Kreative Kudi AI Assistant" width="800">
</p>
