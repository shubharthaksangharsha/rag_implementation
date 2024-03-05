# Apsara AI - Rag & Advance Chatbot 

Welcome to Apsara AI, an innovative project powered by Streamlit, Groq, and Ollama! 🚀

Apsara AI features a cutting-edge Chatbot and RAG (Retrieval Augmentation Generation) system designed to enhance conversational experiences and streamline information retrieval from various documents.

## Features

- **Chatbot:** Engage in real-time conversations with Apsara using advanced conversational capabilities powered by Groq and Ollama.

- **RAG (Retrieval Augmentation Generation):** Extract information from PDFs, Docs, or websites effortlessly with Apsara's RAG system.

## Project Files
- [home.py](https://github.com/shubharthaksangharsha/rag_implementation/blob/main/streamlit_app/home.py): Contains the home page (used this for running the interface) 
- [chatbot.py](https://github.com/shubharthaksangharsha/rag_implementation/blob/main/streamlit_app/pages/chatbot.py): Contains the code for the Chatbot component of Apsara AI.
- [rag.py](https://github.com/shubharthaksangharsha/rag_implementation/blob/main/streamlit_app/pages/rag.py): Contains the code for the RAG (Retrieval Augmentation Generation) component of Apsara AI.

## Getting Started

## Installation Instructions

### Ollama

1. **Download Ollama:**
   - Visit [Ollama's official website](https://ollama.com/download) to download the Ollama software.

2. **Install Ollama:**
   - Follow the instructions provided on the Ollama website to install the software on your system.

3. **Pull Local Embeddings:**
   - To pull the local embeddings, run the following command in your terminal:
     ```bash
     ollama pull nomic-embed-text
     ```

4. **Pull Local LLM (Optional):**
   - If you want to use a local LLM, run the following command in your terminal:
     ```bash
     ollama pull openchat
     ```

5. **Verify Installation:**
   - After installation, you can verify that Ollama and the required components are installed correctly by running:
     ```bash
     ollama --version
     ```
6. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/apsara-ai.git

7. **Install requirements.txt:**
  - Install requirements.txt file by following command 
  ```python
    pip install -r requirements.txt
```
8. **Run the Streamlit App:**
  ```python
    streamlit run home.py
```



