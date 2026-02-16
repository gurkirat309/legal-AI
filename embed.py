import utils as Utils
import os as OS
import google.generativeai as genai
from tqdm import tqdm
import requests
import fitz  # PyMuPDF
import chromadb

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

def split_text_into_sections(text, min_chars_per_section):
    paragraphs = text.split('\n')
    sections = []
    current_section = ""
    current_length = 0

    for paragraph in paragraphs:
        paragraph_length = len(paragraph)
        if current_length + paragraph_length + 2 <= min_chars_per_section:  # +2 for the double newline
            current_section += paragraph + '\n\n'
            current_length += paragraph_length + 2  # +2 for the double newline
        else:
            if current_section:
                sections.append(current_section.strip())
            current_section = paragraph + '\n\n'
            current_length = paragraph_length + 2  # +2 for the double newline

    if current_section:  # Add the last section
        sections.append(current_section.strip())

    return sections

def embed_text_in_chromadb(text, document_name, document_description, persist_directory=Utils.DB_FOLDER):
    # Try to read GEMINI_API_KEY using utils helper (env, .env, export-style)
    api_key = Utils.load_gemini_key()
    if not api_key:
        raise EnvironmentError("Please set GEMINI_API_KEY environment variable or add it to a .env file")

    genai.configure(api_key=api_key)

    documents = split_text_into_sections(text, 1000)

    # generate embeddings with Google Generative Embeddings (use available model)
    embeddings = []
    for d in documents:
        try:
            res = genai.embed_content(model="models/text-embedding-004", content=d)
            emb = None
            if isinstance(res, dict):
                emb = res.get("embedding") or (res.get("data", [{}])[0].get("embedding"))
            else:
                emb = getattr(res, "embedding", None)
                if emb is None and getattr(res, "data", None):
                    emb = getattr(res, "data")[0].get("embedding") if isinstance(getattr(res, "data")[0], dict) else getattr(res, "data")[0].embedding

            if emb is None:
                raise RuntimeError("Embedding not returned by API")

            embeddings.append(emb)
        except Exception as e:
            print(f"Embedding error: {e}")
            embeddings.append([])

    ids = [str(i) for i in range(len(documents))]

    # Metadata for the documents
    metadata = {
        "name": document_name,
        "description": document_description
    }
    metadatas = [metadata] * len(documents)  # Duplicate metadata for each chunk
    
    client = chromadb.PersistentClient(path=persist_directory)
    collection_name = 'collection_1'
    collection = client.get_or_create_collection(name=collection_name)

    # create ids from the current count
    count = collection.count()
    print(f"Collection already contains {count} documents")
    ids = [str(i) for i in range(count, count + len(documents))]

    # load the documents in batches of 100
    for i in tqdm(
        range(0, len(documents), 100), desc="Adding documents", unit_scale=100
    ):
        collection.add(
            ids=ids[i : i + 100],
            documents=documents[i : i + 100],
            metadatas=metadatas[i : i + 100],  # type: ignore
            embeddings=embeddings[i : i + 100],
        )

    new_count = collection.count()
    print(f"Added {new_count - count} documents")
   
if __name__ == "__main__":
    # pdf_path = "TA-9-2024-0138_EN.pdf"
    document_name = "Artificial Intelligence Act"
    document_description = "Artificial Intelligence Act"
    text = pdf_to_text(Utils.EUROPEAN_ACT_URL)
    embed_text_in_chromadb(text, document_name, document_description)

