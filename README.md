# 📚 Rulebook QA System

<img width="1802" height="822" alt="Rulebook QA System" src="https://github.com/user-attachments/assets/5c8bf327-c5d3-465d-87a8-72513af48a64" />

An evidence-first **RAG-based University Rulebook Question Answering System** that retrieves relevant rules, detects conflicts and unsupported topics, and uses an LLM to generate grounded answers.

The system is designed to prioritize **reliability and evidence over guessing**.

---

## ✨ Features

- 🔎 **Semantic-style document retrieval** using TF-IDF and cosine similarity
- 📖 Evidence-based answers from the university rulebook
- 🤖 Groq LLM integration for natural-language responses
- ⚠️ Automatic **conflict detection** when contradictory rules are found
- 🚫 **Not-covered detection** for questions outside the rulebook
- 📌 Section-based citations for retrieved evidence
- 🛡️ Evidence-first prompting to reduce hallucinations
- ⚡ FastAPI backend
- 🌐 Interactive web frontend
- ☁️ Deployed backend using Render
- 🔐 API keys protected using environment variables

---

## 🏗️ Architecture

```text
                    User Question
                          │
                          ▼
                 ┌─────────────────┐
                 │   FastAPI API   │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │  TF-IDF Search  │
                 │ + Cosine Sim.   │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Relevant Rules  │
                 │   Retrieved     │
                 └────────┬────────┘
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
        Coverage Check       Conflict Check
                │                   │
                └─────────┬─────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │   Groq LLM      │
                 │ Grounded Prompt │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Answer + Source │
                 │    Citation     │
                 └─────────────────┘