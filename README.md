# Legal AI
A chatbot to help you navigate through the complicated paths of the AI regulations inside EU

Technically it is a RAG system implementation, using:
- LLM - gemini-2.5-flash
- VectorDB - ChromaDB
- Embedding functions - Gemini (Google Generative AI)
- Agents - LangChain 

It demonstrates how efficient this type of system could be for big documents as a context and how smart the LLM is on understanding legal terms.

For the purpose of this demo, the context is The Artificial Intelligence Act, document adopted by EU Parliament on 13 March 2024. The system could be easily extended to many other legal papers. 

### Installing
After cloning the repository the Gemini API key needs to be added as an environmental variable with the name GEMINI_API_KEY.
```bash
export GEMINI_API_KEY=your_key_value_here
```
This project uses Gemini models: **`gemini-2.5-flash`** for chat and **`textembedding-gecko-001`** for embeddings. After that it should be all fine. To run it locally, in the app folder use:
```bash
streamlit run app.py
```

### Demo
https://huggingface.co/spaces/firica/legalai
