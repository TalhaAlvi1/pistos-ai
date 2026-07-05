<div align="center">

# 🔒 Pistos — Private Local RAG Chatbot

**A 100% local, offline Retrieval-Augmented Generation chatbot that answers only from your own documents.**

No cloud APIs. No data leaves your machine. Powered by Llama 3.2, ChromaDB, and Streamlit.

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/Orchestration-LangChain-1C3C3C)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-6A4C93)](https://www.trychroma.com/)
[![Llama 3.2](https://img.shields.io/badge/LLM-Llama%203.2-0467DF?logo=meta&logoColor=white)](https://huggingface.co/meta-llama)
[![License](https://img.shields.io/badge/License-Private%20Use-lightgrey)](#license)

</div>

---

## 📖 Overview

Pistos is a domain-locked AI chatbot that answers questions **exclusively** from documents you upload — it never falls back on the model's general training knowledge. Everything runs locally: document parsing, embeddings, vector search, and inference. Nothing is sent to an external API.

It was built for theological research use cases (Bible + notes), but the RAG pipeline is fully generic and config-driven — point it at any domain's PDFs, DOCX, TXT, or Markdown files.

<img width="1755" height="968" alt="Screenshot 2026-04-30 200640" src="https://github.com/user-attachments/assets/f589d3b9-7ed4-459d-9f42-3deb9d0c18fe" />


---

## ✨ Features

- **Strict context adherence** — answers only from retrieved chunks, never from the LLM's prior knowledge
- **Fully private & local** — embeddings, vector search, and inference all run on-device
- **Dual LLM backends** — GGUF via `llama-cpp-python` or 4-bit quantized Transformers, swappable per hardware
- **Real-time ingestion progress** — live status for loading, extraction, chunking, and embedding
- **Password-protected admin panel** — upload, manage, and rebuild the vector store via a hidden `?admin=true` route
- **Multi-format ingestion** — PDF, DOCX, TXT, and Markdown
- **Configurable retrieval** — top-k, similarity threshold, chunk size/overlap, and source priority all set in `config.yaml`
- **Configurable persona & tone** — response tone, length, and system prompt are editable without touching code
- **Runs on modest hardware** — CPU-only by default, optional GPU offload

---

## 🏗️ Architecture

<img width="752" height="292" alt="image" src="https://github.com/user-attachments/assets/fb63a658-6da9-4d43-909a-486bc53d97ab" />


**Request flow:** a question is embedded, matched against the vector store, and only the retrieved chunks are passed to the LLM as context — enforced by a strict system prompt that forbids answering outside that context.

---



## 🧰 Tech stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB |
| LLM | Llama 3.2 3B Instruct (GGUF, via `llama-cpp-python`) or Transformers + `bitsandbytes` |
| Document parsing | `pypdf`, `pdfplumber`, `python-docx` |

---

## 🚀 Quick start

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

### 2. Download a model

Download a GGUF build of Llama 3.2 3B Instruct (e.g. from [bartowski/Llama-3.2-3B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF)) and place it at the path set in `config.yaml` under `model.model_path`.

### 3. Set an admin password

```bash
python generate_admin_password.py
```

Copy the generated hash into a `.env` file as instructed by the script.

### 4. Run

```bash
streamlit run app.py
```

| Route | Purpose |
|---|---|
| `http://localhost:8501` | Public chat interface |
| `http://localhost:8501/?admin=true` | Admin panel (upload & manage documents) |

---

## 🖥️ Usage

**Public chat** — ask questions, get answers grounded strictly in the uploaded documents, with source references.

**Admin panel**
1. Navigate to `?admin=true` and log in.
2. Upload PDF / DOCX / TXT / MD files.
3. Watch live ingestion progress:

   ```
   📄 [1/1] Loading: document.pdf (47.1 MB)
   📖 [1/1] Extracting text from PDF
   🔢 [1/1] Creating 1234 embeddings
   ✓ [1/1] Completed (1234 chunks)
   ```
4. Rebuild the vector database on demand.

<img width="1717" height="831" alt="Screenshot 2026-04-30 200715" src="https://github.com/user-attachments/assets/2005b379-0f59-44c2-9710-02f7d6860b18" />
<img width="1048" height="923" alt="Screenshot 2026-05-03 201410" src="https://github.com/user-attachments/assets/bd5bec3c-648f-4b48-9cf5-1ce94b551eb6" />

---

## ⚙️ Configuration

All behavior is controlled through `config.yaml` — no code changes needed:

| Section | Controls |
|---|---|
| `model` | Model path, context size, GPU layers, thread count |
| `document_processing` | Chunk size, overlap, source priority, supported extensions |
| `vectorstore` | Embedding model, top-k, similarity threshold |
| `response` | Fallback message, tone, length, citation formatting |
| `rag` | Strict context adherence, temperature, top-p, max tokens |
| `web_interface` | Host, port, SSL |

---

## 📁 Project structure

```
├── app.py                       # Streamlit entry point
├── config.yaml                  # Central configuration
├── requirements.txt
├── generate_admin_password.py   # Admin password hash generator
├── download_models.py           # Model download helper
├── update_database.py           # Manual vector store rebuild
├── src/
│   ├── web_interface.py         # Streamlit UI + admin panel
│   ├── rag_pipeline.py          # Retrieval + generation orchestration
│   ├── document_processor.py    # Parsing, chunking
│   ├── vector_database.py       # ChromaDB operations
│   ├── llm_manager_gguf.py      # llama-cpp-python backend
│   └── llm_manager_transformers.py  # Transformers backend
└── data/
    ├── uploads/                 # Uploaded source documents
    ├── vectorstore/             # Persisted ChromaDB store
    └── models/                  # Local GGUF model files
```

---

## 📋 Requirements

- Python 3.8+
- 8 GB RAM minimum (16 GB recommended)
- ~5 GB disk space for the model

---


## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---
