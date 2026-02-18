import utils as Utils
import os as OS
from tqdm import tqdm
import requests
import fitz  # PyMuPDF
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

def pdf_to_text(url):
    try:
        response = requests.get(url)
        pdf_data = response.content
        document = fitz.open(stream=pdf_data, filetype="pdf")
        text = ""
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""

def split_text_into_sections(text, chunk_size):
    # Simple split tool to avoid huge chunks
    paragraphs = text.split('\n\n')
    sections = []
    current_chunk = ""
    for p in paragraphs:
        if len(current_chunk) + len(p) < chunk_size:
            current_chunk += p + "\n\n"
        else:
            if current_chunk:
                sections.append(current_chunk.strip())
            current_chunk = p + "\n\n"
    if current_chunk:
        sections.append(current_chunk.strip())
    return sections

def embed_text_in_chromadb(text, document_name, document_description, persist_directory=Utils.DB_FOLDER):
    api_key = Utils.load_gemini_key()
    if not api_key:
        raise EnvironmentError("Please set GEMINI_API_KEY environment variable or add it to a .env file")

    OS.environ["GOOGLE_API_KEY"] = api_key
    
    # Use the EXACT SAME embedding model as agent.py
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    sections = split_text_into_sections(text, 1000)
    
    # Convert to LangChain documents
    documents = [
        Document(
            page_content=section, 
            metadata={"name": document_name, "description": document_description, "chunk": i}
        ) 
        for i, section in enumerate(sections)
    ]

    print(f"Creating vector store in {persist_directory}...")
    
    # Create the DB
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name='legal_collection'
    )
    
    print(f"Successfully embedded {len(documents)} document chunks.")

if __name__ == "__main__":
    document_name = "Artificial Intelligence Act"
    document_description = "Artificial Intelligence Act"
    text = pdf_to_text(Utils.EUROPEAN_ACT_URL)
    if text:
        embed_text_in_chromadb(text, document_name, document_description)
    else:
        print("Failed to extract text from PDF.")

